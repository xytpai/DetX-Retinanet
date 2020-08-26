#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")



at::Tensor anchor_scatter_cuda(const at::Tensor &anchors,  const int batch_size,
						          const int ph, const int pw, const int stride, const float to_move);
at::Tensor anchor_scatter(const at::Tensor &anchors,  const int batch_size,
						          const int ph, const int pw, const int stride, const float to_move)
{
  CHECK_CUDA(anchors);
  return anchor_scatter_cuda(anchors, batch_size, ph, pw, stride, to_move);
}



at::Tensor center_scatter_cuda(const at::Tensor &anchors,  const int batch_size,
						          const int ph, const int pw, const int stride, const float to_move);
at::Tensor center_scatter(const at::Tensor &anchors,  const int batch_size,
						          const int ph, const int pw, const int stride, const float to_move)
{
  CHECK_CUDA(anchors);
  return center_scatter_cuda(anchors, batch_size, ph, pw, stride, to_move);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("anchor_scatter", &anchor_scatter, "anchor_scatter");
  m.def("center_scatter", &center_scatter, "center_scatter");
}
