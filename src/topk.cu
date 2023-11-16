#include <nvtx3/nvToolsExt.h>
// #include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <thread>

#include "assert.h"
#include "topk.h"
#include "unistd.h"

typedef uint4 group_t;  // uint32_t
#define CHECK(res)          \
  if (res != cudaSuccess) { \
    exit(-1);               \
  }
#define GROUP_SIZE 8

// device A100
//  cpu sort :
//  yuan trust sort L: 3002 ms
//  yuan trust sort L: 2750 ms

//  11.16
// 2 threads
// init cost 895 ms
// topk cost 962 ms

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint16_t *doc_lens,
    const size_t n_docs, const uint16_t *query, const int query_len,
    float *scores, int *d_index) {
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
__global__ void pre_process_global(const uint16_t *temp_docs, uint16_t *d_docs,
                                   const uint16_t *d_doc_lens,
                                   const size_t n_docs,
                                   const uint32_t *d_doc_sum) {
  register auto group_sz = sizeof(group_t) / sizeof(uint16_t);
  register auto layer_0_stride = n_docs * group_sz;
  register auto layer_1_stride = group_sz;

  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;
  register auto tidy = blockIdx.y * blockDim.y + threadIdx.y,
                tnumy = gridDim.y * blockDim.y;
  // #pragma unroll
  for (auto i = tidx; i < n_docs; i += tnumx) {
    register auto layer_1_offset = i;
    register auto layer_1_total_offset = layer_1_offset * layer_1_stride;
    // #pragma unroll
    for (auto j = tidy; j < d_doc_lens[i]; j += tnumy) {
      register auto layer_0_offset = j / group_sz;
      register auto layer_2_offset = j % group_sz;
      register auto final_offset = layer_0_offset * layer_0_stride +
                                   layer_1_total_offset + layer_2_offset;
      d_docs[final_offset] = temp_docs[d_doc_sum[i] + j];
    }
  }
}

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs,
                 uint32_t *h_docs_vec, size_t start_idx, size_t lens) {
// h_docs_vec[0] = 0;
#pragma unroll
  for (size_t i = start_idx; i < lens; i++) {
    auto doc_size = docs[i].size();
    // h_docs_vec[i + 1] = h_docs_vec[i] + doc_size;
    register int doc_id = h_docs_vec[i];

#pragma unroll
    for (size_t j = 0; j < doc_size; j++) {
      h_docs[doc_id + j] = docs[i][j];
    }
  }
}

void pre_process_global_thread(std::vector<std::vector<uint16_t>> &docs,
                               uint16_t *h_docs, uint32_t *h_docs_vec,
                               size_t start_idx, size_t lens) {
// h_docs_vec[0] = 0;
#pragma unroll
  for (size_t i = start_idx; i < lens; i++) {
    auto doc_size = docs[i].size();
    // h_docs_vec[i + 1] = h_docs_vec[i] + doc_size;

#pragma unroll
    for (size_t j = 0; j < doc_size; j++) {
      h_docs[h_docs_vec[i] + j] = docs[i][j];
    }
  }
}

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  //   std::chrono::high_resolution_clock::time_point t1 =
  //       std::chrono::high_resolution_clock::now();

  size_t n_docs = docs.size();
  int block = N_THREADS_IN_ONE_BLOCK;
  int grid = (n_docs + block - 1) / block;
  int querys_len = querys.size();

  int *d_sort_index = nullptr;
  float *d_sort_scores = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  uint16_t *d_docs = nullptr;
  uint16_t *d_doc_lens = nullptr;
  // nvtxRangePushA("new *");
  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
  uint32_t *h_docs_vec = new uint32_t[n_docs + 1];
  h_docs_vec[0] = 0;
#pragma unroll_completely
  for (size_t i = 0; i < n_docs; i++) {
    h_docs_vec[i + 1] = h_docs_vec[i] + lens[i];
  }
  //   std::chrono::high_resolution_clock::time_point t4 =
  //   std::chrono::high_resolution_clock::now();
  //   std::cout
  //       << "mallloc cost "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t4 -
  //       t1).count()
  //       << " ms " << std::endl;

  // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);

  std::vector<std::vector<int>> indices_pre(querys_len, std::vector<int>(TOPK));

  const size_t num_threads =
      std::thread::hardware_concurrency() == 0
          ? sysconf(_SC_NPROCESSORS_ONLN) - 1
          : std::thread::hardware_concurrency() / 2;  // 获取硬件支持的线程数
  // num_threads = 2;
  std::vector<std::thread> threads(num_threads);
  size_t chunk_size = n_docs / num_threads;  // 分块大小
  for (size_t i = 0; i < num_threads; i++) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? n_docs : start + chunk_size;
    threads[i] = std::thread(pre_process, std::ref(docs), h_docs, h_docs_vec,
                             start, end);
  }

  // nvtxRangePop();
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  cudaSetDevice(0);

  dim3 numBlocks(32, 32);
  dim3 threadsPerBlock(32, 32);

  cudaStream_t *streams;
  // nvtxRangePushA("streams create");
  streams = (cudaStream_t *)malloc(querys_len * sizeof(cudaStream_t));

  // nvtxRangePop();
  cudaMalloc(&d_sort_index, sizeof(int) * n_docs);
  cudaMalloc(&d_sort_scores, sizeof(float) * n_docs);

  cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  cudaMalloc(&d_doc_lens, sizeof(uint16_t) * n_docs);
  uint32_t *d_doc_sum = nullptr;
  cudaMalloc(&d_doc_sum, sizeof(uint32_t) * (n_docs + 1));
  cudaMemcpy(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_doc_sum, h_docs_vec, sizeof(uint32_t) * (n_docs + 1),
             cudaMemcpyHostToDevice);

  for (std::thread &t : threads) {
    t.join();  // 等待所有线程完成
  }

  //   std::chrono::high_resolution_clock::time_point t3 =
  //   std::chrono::high_resolution_clock::now();
  //   std::cout
  //       << "t_pre_1 cost "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t3 -
  //       t1).count()
  //       << " ms " << std::endl;
  // nvtxRangePushA("temp_docs cp");
  // std::thread pre_process_global_thread(pre_process_global_thread, );
  uint16_t *temp_docs = nullptr;
  cudaMalloc(&temp_docs, sizeof(uint16_t) * h_docs_vec[n_docs]);
  cudaMemcpy(temp_docs, h_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
             cudaMemcpyHostToDevice);
  // nvtxRangePop();
  // nvtxRangePushA("pre_process_global cudaStreamSynchronize");

  // nvtxRangePop();
  // nvtxRangePushA("pre_process_global start");
  pre_process_global<<<numBlocks, threadsPerBlock>>>(
      temp_docs, d_docs, d_doc_lens, n_docs, d_doc_sum);

  // nvtxRangePop();

  // std::chrono::high_resolution_clock::time_point t2 =
  // std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "init cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //     t1).count()
  //     << " ms " << std::endl;

  for (int i = 0; i < querys_len; ++i) {
    // init indices
    // nvtxRangePushA("Loop start");
    cudaStreamCreate(&streams[i]);
    uint16_t *d_query = nullptr;
    float *d_scores = nullptr;
    int *s_indices = nullptr;

    auto &query = querys[i];
    const size_t query_len = query.size();
    // nvtxRangePushA("cuda malloc");
    cudaMallocAsync(&d_scores, sizeof(float) * n_docs, streams[i]);
    cudaMallocAsync(&s_indices, sizeof(int) * n_docs, streams[i]);
    cudaMallocAsync(&d_query, sizeof(uint16_t) * query_len, streams[i]);
    cudaMemcpyAsync(d_query, query.data(), sizeof(uint16_t) * query_len,
                    cudaMemcpyHostToDevice, streams[i]);
    // nvtxRangePop();

    // nvtxRangePushA("topk kernal");
    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores, s_indices);
    // nvtxRangePop();

    // nvtxRangePushA("sort_by_key");
    if (i == 0) {
      cub::DeviceRadixSort::SortPairsDescending(
          d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores,
          s_indices, d_sort_index, n_docs);
      // Allocate temporary storage
      cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams[i]);
    }
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
        d_sort_index, n_docs);
    // nvtxRangePop();

    cudaMemcpyAsync(indices_pre[i].data(), d_sort_index, sizeof(int) * TOPK,
                    cudaMemcpyDeviceToHost, streams[i]);

    cudaFreeAsync(s_indices, streams[i]);
    cudaFreeAsync(d_scores, streams[i]);
    cudaFreeAsync(d_query, streams[i]);
    // nvtxRangePop();
  }
  indices = indices_pre;
  // deallocation
  // cudaFree(d_docs);
  // cudaFree(d_doc_lens);
  // free(h_docs);
}