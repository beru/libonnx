#include <onnx.h>
#include "float16.h"

static int GlobalMaxPool_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int GlobalMaxPool_exit(onnx_node_t* n)
{
	return 1;
}

static int GlobalMaxPool_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	for (i = 0; i < ndim; i++) {
		if (i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
static void GlobalMaxPool_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < C; ++j) {
			o = i * C + j;
			py[o] = px[o * m];
			for (k = 1; k < m; ++k)
				py[o] = max(py[o], px[o * m + k]);
		}
	}
}

void resolver_default_op_GlobalMaxPool(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = GlobalMaxPool_generic<float16_t>,
			.float32_ = GlobalMaxPool_generic<float>,
			.float64_ = GlobalMaxPool_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = GlobalMaxPool_init;
		n->exit = GlobalMaxPool_exit;
		n->reshape = GlobalMaxPool_reshape;
	}
}
