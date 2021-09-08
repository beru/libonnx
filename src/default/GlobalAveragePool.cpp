#include <onnx.h>
#include "refnd.h"
#include "util.h"

namespace onnx {

namespace {

int GlobalAveragePool_reshape(node_t* n)
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
void GlobalAveragePool_generic(node_t* n)
{
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	int N = y->dims[0];
	int C = y->dims[1];
	int avgsz = x->ndata / (N * C);
	std::vector<T> buf(N * C);
	ref2d<T> sum(C, &buf[0]);
	int idx[2], cnt;
	size_t i, j, l;

	for (i = 0, l = x->ndata; i < l; i++) {
		cnt = i;
		idx[0] = cnt / x->strides[0];
		cnt %= x->strides[0];
		idx[1] = cnt / x->strides[1];
		cnt %= x->strides[1];
		sum[idx[0]][idx[1]] += px[i];
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < C; j++)
			py[i * C + j] = sum[i][j] / avgsz;
	}
}

GEN_HOLEDR_TYPE(holder, GlobalAveragePool_generic)

} // namespace

void resolver_default_op_GlobalAveragePool(node_t* n)
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
		n->reshape = GlobalAveragePool_reshape;
	}
}

} // namespace onnx
