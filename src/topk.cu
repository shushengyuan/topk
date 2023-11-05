#include <thread>
#include<omp.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "topk.h"

typedef uint4 group_t;  // uint32_t
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  


void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint32_t query_on_shm[MAX_QUERY_SIZE];
  
#pragma unroll
  for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
    
    query_on_shm[i] = query[i];  // 不太高效的查询加载，假设它不是热点
  }

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;

    for (auto i = 0; i < MAX_DOC_SIZE / GROUP_SIZE; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        int left = query_idx;
        int right = query_len - 1;
        int mid;
        while (left <= right) {
          mid = (left + right) >> 1;
          if (query_on_shm[mid] < doc_segment[j]) {
            left = mid + 1;
          } else {
            right = mid - 1;
          }
        }
        query_idx = left;  // update the query index

        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
  }
}

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs,
                 std::vector<int> &h_doc_lens_vec) {
  auto n_docs = docs.size();

  constexpr auto group_sz = sizeof(group_t) / sizeof(uint16_t);
  auto layer_0_stride = n_docs * group_sz;
  constexpr auto layer_1_stride = group_sz;
auto numProcs = omp_get_num_procs() ;

omp_set_num_threads(8);
#pragma omp parallel
{
#pragma omp for 
  for (int i = 0; i < docs.size(); i++) {
    auto layer_1_offset = i;
    auto layer_1_total_offset = layer_1_offset * layer_1_stride;
    for (int j = 0; j < docs[i].size(); j++) {
      auto layer_0_offset = j / group_sz;

      auto layer_2_offset = j % group_sz;
      auto final_offset = layer_0_offset * layer_0_stride +
                          layer_1_total_offset + layer_2_offset;
      h_docs[final_offset] = docs[i][j];
    }
    h_doc_lens_vec[i] = docs[i].size();
  }
  }
}

  void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  auto n_docs = docs.size();
  // float *d_scores = nullptr;
  uint16_t *d_docs = nullptr;
  // uint16_t *d_query = nullptr;
  int *d_doc_lens = nullptr;

  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];

  std::vector<int> h_doc_lens_vec(n_docs);

  std::thread t1(pre_process, std::ref(docs), h_docs, std::ref(h_doc_lens_vec));

  cudaStream_t stream = cudaStreamPerThread;
  // copy to device
  cudaMallocAsync(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, stream);
  // cudaMallocAsync(&d_scores, sizeof(float) * n_docs, stream);
  cudaMallocAsync(&d_doc_lens, sizeof(int) * n_docs, stream);

  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);

  cudaSetDevice(0);

  int block = N_THREADS_IN_ONE_BLOCK;
  int grid = (n_docs + block - 1) / block;
  int querys_len = querys.size();
  // float ** scores_group;

auto numProcs = omp_get_num_procs() ;
// std::cout<<numProcs<<std::endl;
omp_set_num_threads(8);
// int *host_indices= new int[n_docs]; // why
std::vector<int> host_indices(n_docs); // why

// #pragma omp parallel
//  {
// #pragma omp for 
  for (int i = 0; i < n_docs; ++i) {
    host_indices[i] = i;
  }
  // }
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < querys_len; i++) {
    
    cudaStreamCreate(&streams[i]);

  }  
  t1.join();
  cudaMemcpyAsync(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
                  cudaMemcpyHostToDevice, stream);
 
  for (int i = 0; i < querys_len; ++i) {
    // init indices
    uint16_t *d_query = nullptr;
    float *d_scores = nullptr;
    int *s_indices= nullptr;

    auto &query = querys[i];
    const size_t query_len = query.size();
    
    cudaMallocAsync(&d_query, sizeof(uint16_t) * query_len, streams[i]);
    cudaMemcpyAsync(d_query, query.data(), sizeof(uint16_t) * query_len,
                    cudaMemcpyHostToDevice, streams[i]);
    cudaMallocAsync(&d_scores, sizeof(float) * n_docs, streams[i]);
    cudaMallocAsync(&s_indices, sizeof(int) * n_docs, streams[i]);
    cudaMemcpyAsync(s_indices, host_indices.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice, streams[i]);

  docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores);
    
        cudaFreeAsync(d_query, streams[i]);
        // cudaMemcpyAsync(scores,d_scores, sizeof(float) * n_docs,
        // cudaMemcpyDeviceToHost, streams[i]);
        thrust::device_ptr<float> scores_key(d_scores);
        thrust::device_ptr<int> s_indices_value(s_indices);
        thrust::sort_by_key(scores_key, scores_key + n_docs, s_indices_value,thrust::greater<float>());
       

        std::vector<int> host_indices_temp(TOPK); // why
        cudaMemcpyAsync(host_indices_temp.data(), s_indices, sizeof(int) * TOPK, cudaMemcpyDeviceToHost, streams[i]);
        cudaFreeAsync(s_indices, streams[i]);
        cudaFreeAsync(d_scores, streams[i]);
        cudaFreeAsync(d_query, streams[i]);
      indices.push_back(host_indices_temp);
      
    }
    // cudaDeviceSynchronize();
  


  
   
    // cudaFreeAsync(d_query, stream);


  // deallocation
  // cudaFree(d_docs);
  // cudaFreeAsync(d_query);
  // cudaFree(d_scores);
  // cudaFree(d_doc_lens);
  // free(h_docs);
}
