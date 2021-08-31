#include <onnx.h>
#include "util.h"

namespace {

struct ope_pdata_t : public onnx_node_t::ope_pdata_t {
	std::vector<int> perm;
};

bool Transpose_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->perm.resize(n->inputs[0]->ndim);
	int64_t* ints;
	if (pdat->perm.size() == n->read_attribute("perm", &ints)) {
		for (int i = 0; i < pdat->perm.size(); i++)
			pdat->perm[i] = ints[i];
	}else {
		for (int i = 0; i < pdat->perm.size(); i++)
			pdat->perm[i] = pdat->perm.size() - i - 1;
	}
	n->priv = pdat;
	return true;
}

int Transpose_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	if (y->reshape_identity(x)) {
		for (int i = 0; i < x->ndim; i++)
			y->dims[i] = x->dims[pdat->perm[i]];
		return 1;
	}
	return 0;
}

template <typename T>
void Transpose_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	size_t nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t l;

	for (oy = 0, l = y->ndata; oy < l; oy++) {
		y->offset_to_indices(oy, &iy[0]);
		for (size_t i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		py[oy] = px[ox];
	}
}

GEN_HOLEDR_TYPE(holder, Transpose_generic)

} // namespace

void resolver_default_op_Transpose(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bool_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double, bfloat16_t,
			std::complex<float>, std::complex<double>,
			std::string
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			bool_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double,
			std::complex<float>, std::complex<double>,
			std::string
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Transpose_init;
		n->reshape = Transpose_reshape;
	}
}
