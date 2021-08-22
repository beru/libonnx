#include <onnx.h>
#include "float16.h"

struct ope_pdata_t {
	float p;
};

static int GlobalLpPool_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		if (n->opset >= 2)
			pdat->p = n->attribute_read_int("p", 2);
		else
			pdat->p = n->attribute_read_float("p", 2.0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int GlobalLpPool_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int GlobalLpPool_reshape(onnx_node_t* n)
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
static void GlobalLpPool_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
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
			for (k = 0, py[o] = 0; k < m; ++k)
				py[o] += pow(abs(px[o * m + k]), (T)pdat->p);
			py[o] = pow(py[o], T(1.0 / pdat->p));
		}
	}
}

void resolver_default_op_GlobalLpPool(onnx_node_t* n)
{
	if (n->opset >= 2) {
		n->ope = onnx_ope_type_selector{
			.float16_ = GlobalLpPool_generic<float16_t>,
			.float32_ = GlobalLpPool_generic<float>,
			.float64_ = GlobalLpPool_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = GlobalLpPool_generic<float16_t>,
			.float32_ = GlobalLpPool_generic<float>,
			.float64_ = GlobalLpPool_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = GlobalLpPool_init;
		n->exit = GlobalLpPool_exit;
		n->reshape = GlobalLpPool_reshape;
	}
}
