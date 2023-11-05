#include <thread>
#include <omp.h>


#include <cub/cub.cuh> 

#include <nvtx3/nvToolsExt.h>
#include <cstdlib>
#include "topk.h"

using namespace cub;
typedef uint4 group_t;  // uint32_t

// #define CubDebugExit	(e)if(CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)) { exit(1); }

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  

constexpr size_t N = 10;
using type = int64_t;
using value_type = long;
template<typename T>
void print(T array[], size_t N) {
    for(size_t i = 0; i < N; i++) {
        std::cout << float(array[i]) << ", ";
    }
    std::cout << std::endl;
}

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores, int *d_index) {
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
    d_index[doc_id] = doc_id;
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
  std::vector<float> scores(n_docs);
  std::vector<int> s_indices(n_docs);
  // float *d_scores = nullptr;
  uint16_t *d_docs = nullptr;
  // uint16_t *d_query = nullptr;
  int *d_doc_lens = nullptr;

  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];

  std::vector<int> h_doc_lens_vec(n_docs);

  nvtxRangePushA("pre_process");
  std::thread t1(pre_process, std::ref(docs), h_docs, std::ref(h_doc_lens_vec));
  nvtxRangePop();

  cudaStream_t stream = cudaStreamPerThread;
  // copy to device

  nvtxRangePushA("h_docs_decvice_to_host_Malloc");
  cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  nvtxRangePop();

  // cudaMallocAsync(&d_scores, sizeof(float) * n_docs, stream);
  cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

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
#pragma omp parallel
	{
#pragma omp for 
  for (int i = 0; i < n_docs; ++i) {
    s_indices[i] = i;
  }
  }
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < querys_len; i++) {
    
    cudaStreamCreate(&streams[i]);

  }  
  t1.join();

  nvtxRangePushA("h_docs_decvice_to_host_cudaMemcpyAsync");
  cudaMemcpyAsync(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  cudaMemcpyHostToDevice, stream);
  nvtxRangePop();
  
  nvtxRangePushA("h_doc_lens_vec.data()_decvice_to_host_cudaMemcpyAsync");
  cudaMemcpyAsync(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
                  cudaMemcpyHostToDevice, stream);
  nvtxRangePop();

  for (int i = 0; i < querys_len; ++i) {
    // nvtx3::scoped_range range("This is a push/pop range");
    // init indices
    uint16_t *d_query = nullptr;
    float *d_scores = nullptr;
    float *d__sort_scores = nullptr;

    int *d_init_index = nullptr;
    int *d_sort_index = nullptr;
    int *h_sort_index = (int*)malloc(sizeof(int) * TOPK);


    auto &query = querys[i];
    const size_t query_len = query.size();
    

    
    cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
    cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len,
                    cudaMemcpyHostToDevice);
    cudaMalloc(&d_scores, sizeof(float) * n_docs);
    cudaMalloc(&d__sort_scores, sizeof(float) * n_docs);

    cudaMalloc(&d_init_index, sizeof(int) * n_docs);

  // if(i == 0){
   
  // }
  nvtxRangePushA("docQueryScoringCoalescedMemoryAccessSampleKernel");

  docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores, d_init_index);
    
        cudaFreeAsync(d_query, streams[i]);
        nvtxRangePop();

        // scores.data()_decvice_to_host_cudaMemcpyAsync
        nvtxRangePushA("scores.data()_decvice_to_host_cudaMemcpyAsync");
        // cudaMemcpyAsync(scores.data(), d_scores, sizeof(float) * n_docs,
        // cudaMemcpyDeviceToHost, streams[i]);
        nvtxRangePop();


// -----------------cpu sort --------------------------------
        // cudaFreeAsync(d_scores, streams[i]);

      // partial_sort
      // nvtxRangePushA("partial_sort");
      // std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
      //                   s_indices.end(), [&scores](const int &a, const int &b) {
      //                     if (scores[a] != scores[b]) {
      //                       return scores[a] > scores[b];  // 按照分数降序排序
      //                     }
      //                     return a < b;  // 如果分数相同，按索引从小到大排序
      //                   });
      // std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
      // indices.push_back(s_ans);
      // nvtxRangePop();

      // nvtxRangePushA("GPU_topK");
      // uint16_t *my_topk_gpu_scores = nullptr;                  
      // cudaMalloc(&my_topk_gpu_scores, sizeof(uint16_t) * TOPK);              
      // top_k_gpu(d_scores, sizeof(uint16_t) * n_docs, TOPK, my_topk_gpu_scores);
      // nvtxRangePop();

// -----------------gpu sort --------------------------------
      // Allocate host arrays
      // Determine temporary device storage requirements
// SortPairs (void *d_temp_storage, size_t &temp_storage_bytes, const KeyT *d_keys_in, KeyT *d_keys_out, const ValueT *d_values_in, 
// ValueT *d_values_out, NumItemsT num_items, int begin_bit=0, int end_bit=sizeof(KeyT)*8, cudaStream_t stream=0, bool debug_synchronous=false)
      std::cout << "start gpu sort" << std::endl;

      // cudaDeviceSynchronize();
      // nvtxRangePushA("gpu sort");
      // void     *d_temp_storage = NULL;
      // size_t   temp_storage_bytes = 0;
      // cudaStream_t stream=0;
      // // CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

      // // int size 
      // nvtxRangePushA("DeviceRadixSort_1");
      // cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,d_scores, d__sort_scores, d_init_index, d_sort_index, n_docs,0, sizeof(float)*8, stream, true);
      // nvtxRangePop();

      // // Allocate temporary storage
      // cudaMalloc(&d_temp_storage, temp_storage_bytes);
      // cudaDeviceSynchronize();

      // // Run sorting operation
      // nvtxRangePushA("DeviceRadixSort_2");
      // cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,d_scores, d__sort_scores, d_init_index, d_sort_index, n_docs,0, sizeof(float)*8, stream, true);
      // nvtxRangePop();


      // nvtxRangePushA("indices.push_back(host_indices_temp);");
      // // d_sort_index memcpy to the host as the result
      // std::vector<int> host_indices_temp(TOPK);
      // cudaMemcpy(host_indices_temp.data() , d_sort_index, sizeof(int) * TOPK, cudaMemcpyDeviceToHost);

      // // std::vector<int> s_ans(h_sort_index, h_sort_index + TOPK);
      // indices.push_back(host_indices_temp);

      // // g_allocator.DeviceFree(d__sort_scores);
      // // g_allocator.DeviceFree(d_sort_index);
      // // g_allocator.DeviceFree(d_temp_storage);


      // nvtxRangePop();
      // nvtxRangePop();
//  测试用例 这个就能无痛运行 ...
      type *h1, *h2, *d1, *d2;
      value_type *d3, *d4;
      void *tmp;
      h1 = new type[N];
      h2 = new type[N];
      for(size_t i = 0; i < N; i++) {
          h1[i] = -i;
      }
      print(h1, N);

      cudaMalloc(&d1, sizeof(type) * N);
      cudaMalloc(&d2, sizeof(type) * N);
      cudaMalloc(&d3, sizeof(value_type) * N);
      cudaMalloc(&d4, sizeof(value_type) * N);
      cudaMemcpy(d1, h1, sizeof(type) * N, cudaMemcpyDefault);
      size_t temp_storage_bytes1;
      cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes1, d1, d2, d3, d4, N, 0, 10);
      cudaMalloc(&tmp, temp_storage_bytes1);
      cudaDeviceSynchronize();
      cub::DeviceRadixSort::SortPairs(tmp, temp_storage_bytes1, d1, d2, d3, d4, N, 0, 10);
      cudaDeviceSynchronize();
      cudaMemcpy(h2, d2, sizeof(type) * N, cudaMemcpyDefault);
      print(h2, N);

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

