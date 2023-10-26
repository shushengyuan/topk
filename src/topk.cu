#include "topk.h"

typedef uint4 group_t;  // uint32_t
const size_t group_size = 8;

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
  for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
    query_on_shm[i] = query[i];  // 不太高效的查询加载，假设它不是热点
  }

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;

    for (auto i = 0; i < MAX_DOC_SIZE / group_size; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < group_size; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        int left = query_idx;
        int right = query_len - 1;
        int mid;
        while (left <= right) {
          mid = (left + right) / 2;
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

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices, uint16_t *h_docs,
    std::vector<int> &h_doc_lens_vec  // shape [querys.size(), TOPK]
) {
  // printf("uint4 %lu group_t %lu uint16_t %lu rest %lu
  // \n",sizeof(uint4),sizeof(group_t),sizeof(uint16_t),sizeof(group_t) /
  // sizeof(uint16_t));
  auto n_docs = docs.size();
  std::vector<float> scores(n_docs);
  std::vector<int> s_indices(n_docs);
  float *d_scores = nullptr;
  uint16_t *d_docs = nullptr, *d_query = nullptr;
  int *d_doc_lens = nullptr;

  cudaStream_t stream = cudaStreamPerThread;
  // copy to device
  cudaMallocAsync(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, stream);
  cudaMallocAsync(&d_scores, sizeof(float) * n_docs, stream);
  cudaMallocAsync(&d_doc_lens, sizeof(int) * n_docs, stream);

  cudaMemcpyAsync(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
                  cudaMemcpyHostToDevice, stream);

  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);

  cudaSetDevice(0);

  int block = N_THREADS_IN_ONE_BLOCK;
  int grid = (n_docs + block - 1) / block;
#pragma unroll
  for (int i = 0; i < n_docs; ++i) {
    s_indices[i] = i;
  }
  int index = 0;
  for (auto &query : querys) {
    // init indices

    const size_t query_len = query.size();
    cudaMallocAsync(&d_query, sizeof(uint16_t) * query_len, stream);
    cudaMemcpyAsync(d_query, query.data(), sizeof(uint16_t) * query_len,
                    cudaMemcpyHostToDevice, stream);
    // launch kernel

    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       stream>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores);

    if (index++ != 0) {
      std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
                        s_indices.end(), [&scores](const int &a, const int &b) {
                          if (scores[a] != scores[b]) {
                            return scores[a] > scores[b];  // 按照分数降序排序
                          }
                          return a < b;  // 如果分数相同，按索引从小到大排序
                        });
      std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
      indices.push_back(s_ans);
    }

    cudaMemcpyAsync(scores.data(), d_scores, sizeof(float) * n_docs,
                    cudaMemcpyDeviceToHost, stream);
  }

  // cudaStreamSynchronize(stream);
  std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
                    s_indices.end(), [&scores](const int &a, const int &b) {
                      if (scores[a] != scores[b]) {
                        return scores[a] > scores[b];  // 按照分数降序排序
                      }
                      return a < b;  // 如果分数相同，按索引从小到大排序
                    });
  std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
  indices.push_back(s_ans);

  cudaFreeAsync(d_query, stream);

  // deallocation
  cudaFree(d_docs);
  // cudaFreeAsync(d_query);
  cudaFree(d_scores);
  cudaFree(d_doc_lens);
  // free(h_docs);
}
