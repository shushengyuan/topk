# 接下来的优化思路与分工

# Abstract

* 压缩通信
  * 具备优化的可行性
  * 进阶优化
    * *流水线 pipeline *
    * 进一步节省时间


* cpu操作gpu加速

cpu  to gpu
```cpp
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

```


* batch size 
  * 批量处理batch 运行一次query kernel   N 次sort


* stream 优化


* 真·使用topk
  * 有思路 有讲解 但是没能用的代码
