#include <nvtx3/nvToolsExt.h>
// #include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <thread>

#include "assert.h"
#include "topk.h"

typedef uint4 group_t;  // uint32_t
#define CHECK(res)          \
  if (res != cudaSuccess) { \
    exit(-1);               \
  }

// device A100
//  cpu sort :
//  yuan trust sort L: 3002 ms
//  yuan trust sort L: 2750 ms

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint16_t *doc_lens,
    const size_t n_docs, const uint16_t *d_query, const int bach_now,
    const uint16_t *query_lens_d, float *scores, int *s_indices,
    const uint32_t *d_query_sum) {
  // each thread process one doc-query pair scoring task
  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;

  if (tidx >= n_docs * bach_now) {
    return;
  }
  __shared__ uint32_t query_on_shm[MAX_QUERY_SIZE];
  // __shared__ uint32_t doc_lens_on_shm[n_docs];

  for (auto doc_id = tidx; doc_id < n_docs * bach_now; doc_id += tnumx) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;
    register int q_id = doc_id / n_docs;
    register auto query_len = query_lens_d[q_id];
    register auto start_index = d_query_sum[q_id] - d_query_sum[0];
    register auto doc_index = doc_id % n_docs;

#pragma unroll
    for (auto i = threadIdx.y; i < query_len; i += blockDim.y) {
      query_on_shm[i] =
          d_query[start_index + i];  // not very efficient query loading
      // temporally, as assuming its not
      // hotspot
    }
    __syncthreads();

    for (auto i = 0; i < MAX_DOC_SIZE / GROUP_SIZE; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded =
          ((group_t *)docs)[i * n_docs + doc_index];  // tidx
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        while (query_idx < query_len &&
               query_on_shm[query_idx] < doc_segment[j]) {
          ++query_idx;
        }

        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      __syncwarp();
    }

    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_index]);  // tidx
    s_indices[doc_id] = doc_index;
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

void query_thread(std::vector<std::vector<uint16_t>> &querys, uint16_t *h_query,
                  uint32_t *query_lens_sum) {
#pragma unroll
  for (size_t i = 0; i < querys.size(); i++) {
    auto querys_size = querys[i].size();

#pragma unroll
    for (size_t j = 0; j < querys_size; j++) {
      h_query[query_lens_sum[i] + j] = querys[i][j];
    }
  }
}

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs,
                 uint32_t *h_docs_vec, size_t start_idx, size_t lens) {
#pragma unroll
  for (size_t i = start_idx; i < lens; i++) {
    auto doc_size = docs[i].size();
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
  // std::chrono::high_resolution_clock::time_point t1 =
  //     std::chrono::high_resolution_clock::now();

  size_t n_docs = docs.size();
  int total_querys_len = querys.size();
  const int BATCH_SIZE = total_querys_len;

  int block = N_THREADS_IN_ONE_BLOCK;
  int grid = ((BATCH_SIZE * n_docs) + block - 1) / block;

  uint16_t *d_docs = nullptr;
  uint16_t *d_doc_lens = nullptr;
  uint16_t *temp_docs = nullptr;
  std::vector<uint16_t> query_lens_vec(total_querys_len);
  std::vector<std::vector<int>> indices_pre(querys.size(),
                                            std::vector<int>(TOPK));

  uint32_t *h_docs_vec = new uint32_t[n_docs + 1];
  uint32_t *query_lens_sum = new uint32_t[total_querys_len + 1];
  h_docs_vec[0] = 0;
  query_lens_sum[0] = 0;
#pragma unroll_completely
  for (size_t i = 0; i < n_docs; i++) {
    h_docs_vec[i + 1] = h_docs_vec[i] + lens[i];
  }
#pragma unroll_completely
  for (size_t i = 0; i < total_querys_len; i++) {
    query_lens_vec[i] = querys[i].size();
    query_lens_sum[i + 1] = query_lens_sum[i] + querys[i].size();
  }

  uint16_t *h_docs = new uint16_t[h_docs_vec[n_docs]];
  uint16_t *h_query = new uint16_t[query_lens_sum[total_querys_len]];

  std::thread t_pre_1(pre_process, std::ref(docs), h_docs, h_docs_vec, 0,
                      n_docs / 2);
  std::thread t_pre_2(pre_process, std::ref(docs), h_docs, h_docs_vec,
                      n_docs / 2, n_docs);

  std::thread t_query(query_thread, std::ref(querys), h_query, query_lens_sum);

  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  cudaSetDevice(0);

  dim3 numBlocks(32, 32);
  dim3 threadsPerBlock(32, 32);

  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(total_querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < total_querys_len; i++) {
    cudaStreamCreate(&streams[i]);
  }

  cudaMallocAsync(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  streams[0]);
  uint32_t *d_doc_sum = nullptr;
  uint32_t *d_query_sum = nullptr;
  cudaMallocAsync(&d_doc_sum, sizeof(uint32_t) * (n_docs + 1), streams[0]);
  cudaMemcpyAsync(d_doc_sum, h_docs_vec, sizeof(uint32_t) * (n_docs + 1),
                  cudaMemcpyHostToDevice, streams[0]);
  cudaMallocAsync(&d_query_sum, sizeof(uint32_t) * (total_querys_len + 1),
                  streams[1]);
  cudaMemcpyAsync(d_query_sum, query_lens_sum,
                  sizeof(uint32_t) * (total_querys_len + 1),
                  cudaMemcpyHostToDevice, streams[1]);
  cudaMallocAsync(&temp_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
                  streams[2]);
  cudaMallocAsync(&d_doc_lens, sizeof(uint16_t) * n_docs, streams[1]);
  cudaMemcpyAsync(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
                  cudaMemcpyHostToDevice, streams[1]);
  t_pre_1.join();
  t_pre_2.join();
  cudaMemcpyAsync(temp_docs, h_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
                  cudaMemcpyHostToDevice, streams[2]);

  cudaStreamSynchronize(streams[1]);
  cudaStreamSynchronize(streams[2]);
  // nvtxRangePushA("pre_process_global start");
  pre_process_global<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(
      temp_docs, d_docs, d_doc_lens, n_docs, d_doc_sum);
  // cudaFreeAsync(&d_doc_sum, streams[0]);
  // cudaFreeAsync(&temp_docs, streams[1]);
  // nvtxRangePop();

  // std::chrono::high_resolution_clock::time_point t2 =
  //     std::chrono::high_resolution_clock::now();

  // std::cout
  //     << "init cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //     t1).count()
  // free(h_docs);

  uint16_t *query_lens_d = nullptr;
  cudaMallocAsync(&query_lens_d, sizeof(uint16_t) * total_querys_len,
                  streams[1]);
  cudaMemcpyAsync(query_lens_d, query_lens_vec.data(),
                  sizeof(uint16_t) * total_querys_len, cudaMemcpyHostToDevice,
                  streams[1]);
  cudaStreamSynchronize(streams[0]);
  cudaStreamSynchronize(streams[1]);
  t_query.join();
  auto i = 0;

  while (i < total_querys_len) {
    auto bach_now = BATCH_SIZE;
    if (i + BATCH_SIZE == total_querys_len) {
      bach_now = BATCH_SIZE;
      // break;
    } else if (i + BATCH_SIZE > total_querys_len) {
      bach_now = total_querys_len - i;
    }
    if (bach_now == 0) {
      break;
    }

    // nvtxRangePushA("query start");

    float *d_scores = nullptr;
    int *s_indices = nullptr;
    uint16_t *d_query = nullptr;

    cudaMallocAsync(
        &d_query,
        sizeof(uint16_t) * (query_lens_sum[i + bach_now] - query_lens_sum[i]),
        streams[i]);

    cudaMemcpyAsync(
        d_query, h_query + query_lens_sum[i],
        sizeof(uint16_t) * (query_lens_sum[i + bach_now] - query_lens_sum[i]),
        cudaMemcpyHostToDevice, streams[i]);

    cudaMallocAsync(&d_scores, sizeof(float) * n_docs * bach_now, streams[i]);
    cudaMallocAsync(&s_indices, sizeof(int) * n_docs * bach_now, streams[i]);
    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, bach_now, query_lens_d + i,
        d_scores, s_indices, d_query_sum + i);
    // free(h_query);
    // nvtxRangePop();
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int *d_sort_index = nullptr;
    float *d_sort_scores = nullptr;

    cudaMallocAsync(&d_sort_index, sizeof(int) * n_docs, streams[i]);
    cudaMallocAsync(&d_sort_scores, sizeof(float) * n_docs, streams[i]);

    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
        d_sort_index, n_docs);
    // Allocate temporary storage
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams[i]);
    // cudaDeviceSynchronize();
#pragma unroll
    for (int j = 0; j < bach_now; ++j) {
      // nvtxRangePushA("sort_by_key");
      cub::DeviceRadixSort::SortPairsDescending(
          d_temp_storage, temp_storage_bytes, d_scores + j * n_docs,
          d_sort_scores, s_indices + j * n_docs, d_sort_index, n_docs);

      cudaMemcpyAsync(indices_pre[j + i].data(), d_sort_index,
                      sizeof(int) * TOPK, cudaMemcpyDeviceToHost, streams[j]);
      cudaMemsetAsync(d_sort_index, 0, n_docs * sizeof(int), streams[j]);
      cudaMemsetAsync(d_sort_scores, 0, n_docs * sizeof(float), streams[j]);
      // cudaFree(d_temp_storage);
      // nvtxRangePop();
    }
    // cudaFreeAsync(s_indices, streams[i]);
    // cudaFreeAsync(d_scores, streams[i]);
    // cudaFreeAsync(d_query, streams[i]);

    i += bach_now;
  }

  indices = indices_pre;
  // cudaFreeAsync(s_indices, streams[0]);
  // cudaFreeAsync(d_scores, streams[0]);
  // cudaFreeAsync(d_query, streams[0]);
  // // deallocation
  // cudaFree(d_docs);
  // cudaFree(d_scores);
  // cudaFreeAsync(d_docs, streams[0]);
  // cudaFreeAsync(d_doc_lens, streams[0]);
  // cudaFree(d_doc_lens);
}