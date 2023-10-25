#include "preprocess.h"

typedef uint4 group_t;  // uint32_t

void pre_process(std::vector<std::vector<uint16_t>> &docs, float *d_scores,
                 uint16_t *d_docs, int *d_doc_lens) {
  auto n_docs = docs.size();

  cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  cudaMalloc(&d_scores, sizeof(float) * n_docs);
  cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
  memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  std::vector<int> h_doc_lens_vec(n_docs);

  constexpr auto group_sz = sizeof(group_t) / sizeof(uint16_t);
  auto layer_0_stride = n_docs * group_sz;
  constexpr auto layer_1_stride = group_sz;

  constexpr int layer_0_shift =
      __builtin_ctz(group_sz);  // 计算layer_0_stride是2的多少次方
  // printf("%d \n", layer_0_shift);
  constexpr auto layer_2_mask = group_sz - 1;

  for (int i = 0; i < docs.size(); i++) {
    auto layer_1_offset = i;
    auto layer_1_total_offset = layer_1_offset * layer_1_stride;
#pragma unroll
    for (int j = 0; j < docs[i].size(); j++) {
      auto layer_0_offset = j >> layer_0_shift;

      auto layer_2_offset = j & layer_2_mask;
      auto final_offset = layer_0_offset * layer_0_stride +
                          layer_1_total_offset + layer_2_offset;
      h_docs[final_offset] = docs[i][j];
    }
    h_doc_lens_vec[i] = docs[i].size();
  }

  cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
             cudaMemcpyHostToDevice);

  free(h_docs);
}