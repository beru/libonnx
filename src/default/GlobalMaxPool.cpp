#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

int GlobalMaxPool_reshape(node_t* n)
{
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);

	for (int i = 0; i < ndim; i++) {
		if (i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
void GlobalMaxPool_generic(node_t* n)
{
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < C; ++j) {
			int o = i * C + j;
			py[o] = px[o * m];
			for (int k = 1; k < m; ++k)
				py[o] = max(py[o], px[o * m + k]);
		}
	}
}

GEN_HOLEDR_TYPE(holder, GlobalMaxPool_generic)

} // namespace

void resolver_default_op_GlobalMaxPool(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = [](node_t* n) {
			return is_inout_size(n, 1, 1);
		};
		n->reshape = GlobalMaxPool_reshape;
	}
}

} // namespace onnx
