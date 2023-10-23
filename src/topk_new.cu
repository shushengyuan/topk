// 定义一个常量，表示每个线程束的大小
#define WARP_SIZE 32

// 定义一个函数模板，用于指定块的大小
template <unsigned int blockSize>
__global__ void docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
  // 每个线程处理一个文档-查询对的评分任务
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
  // 使用循环展开来减少分支冲突
#pragma unroll
  for (auto i = threadIdx.x; i < query_len; i += blockSize) {
    query_on_shm[i] = query[i];  // 不太高效的查询加载，假设它不是热点
  }

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;

    // 使用uint4类型来实现分组内存访问，每次读取4个元素
    // typedef uint4 group_t;
    for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t));
         i++) {
      if (no_more_load) {
        break;
      }
      // 使用分组内存访问来提高内存吞吐量
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        // 使用位运算代替求余运算
        while (query_idx < query_len &&
               query_on_shm[query_idx] < doc_segment[j]) {
          ++query_idx;
        }
        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      // 使用__syncwarp()代替__syncthreads()来同步分组中的线程
      __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
  }
}
