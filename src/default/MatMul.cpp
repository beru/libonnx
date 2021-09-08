#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	int m;
	int n;
	int k;
};

bool MatMul_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 2, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->m = 0;
	pdat->n = 0;
	pdat->k = 0;
	n->priv = pdat;
	return true;
}

int MatMul_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	std::vector<int> adims;
	std::vector<int> bdims;

	if (a->ndim == 1) {
		adims.resize(2);
		adims[0] = 1;
		adims[1] = a->dims[0];
	}else {
		adims = a->dims;
	}
	if (b->ndim == 1) {
		bdims[0] = b->dims[0];
		bdims[1] = 1;
	}else {
		bdims = b->dims;
	}
	int ndim = max(adims.size(), bdims.size());
	std::vector<int> dims(ndim);
	if (adims.size() < 2 || bdims.size() < 2)
		return 0;
	if (adims[adims.size() - 1] != bdims[bdims.size() - 2])
		return 0;
	dims[ndim - 2] = adims[adims.size() - 2];
	dims[ndim - 1] = bdims[bdims.size() - 1];
	for (int i = 3; i <= ndim; i++) {
		int alen = (adims.size() - i) < 0 ? 1 : adims[adims.size() - i];
		int blen = (bdims.size() - i) < 0 ? 1 : bdims[bdims.size() - i];
		if (alen != blen && alen > 1 && blen > 1)
			return 0;
		dims[ndim - i] = max(alen, blen);
	}
	pdat->m = adims[adims.size() - 2];
	pdat->n = bdims[bdims.size() - 1];
	pdat->k = adims[adims.size() - 1];
	return y->reshape(&dims[0], ndim, a->type);
}

template <typename T>
void MatMul_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i += pdat->m * pdat->n) {
		T* pa = (T*)a->broadcast_map_address(y, i);
		T* pb = (T*)b->broadcast_map_address(y, i);
		for (int u = 0; u < pdat->m; u++) {
			for (int v = 0; v < pdat->n; v++) {
				T sum = 0;
				for (int w = 0; w < pdat->k; w++)
					sum += pa[u * pdat->k + w] * pb[w * pdat->n + v];
				py[i + u * pdat->n + v] = sum;
			}
		}
	}
}

GEN_HOLEDR_TYPE(holder, MatMul_generic)

} // namespace

void resolver_default_op_MatMul(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = MatMul_init;
		n->reshape = MatMul_reshape;
	}
}
