#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float p;
};

bool GlobalLpPool_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	if (n->opset >= 2)
		pdat->p = n->read_attribute("p", 2);
	else
		pdat->p = n->read_attribute("p", 2.0f);
	n->priv = pdat;
	return true;
}

int GlobalLpPool_reshape(node_t* n)
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
void GlobalLpPool_generic(node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
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
			py[o] = 0;
			for (int k = 0; k < m; ++k)
				py[o] += pow(abs(px[o * m + k]), (T)pdat->p);
			py[o] = pow(py[o], T(1.0 / pdat->p));
		}
	}
}

GEN_HOLEDR_TYPE(holder, GlobalLpPool_generic)

} // namespace

void resolver_default_op_GlobalLpPool(node_t* n)
{
	if (n->opset >= 2) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = GlobalLpPool_init;
		n->reshape = GlobalLpPool_reshape;
	}
}

} // namespace onnx
