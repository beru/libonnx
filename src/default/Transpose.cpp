#include <onnx.h>
#include "util.h"

namespace {

struct ope_pdata_t {
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
	if (pdat->perm.size() == n->attribute_read_ints("perm", &ints)) {
		for (int i = 0; i < pdat->perm.size(); i++)
			pdat->perm[i] = ints[i];
	}else {
		for (int i = 0; i < pdat->perm.size(); i++)
			pdat->perm[i] = pdat->perm.size() - i - 1;
	}
	n->priv = pdat;
	return true;
}

int Transpose_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
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

void Transpose_string(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	char** px = (char**)x->data;
	char** py = (char**)y->data;
	int nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for (oy = 0, l = y->ndata; oy < l; oy++) {
		y->offset_to_indices(oy, &iy[0]);
		for (i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		if (py[oy])
			free(py[oy]);
		py[oy] = strdup(px[ox]);
	}
}

} // namespace

void resolver_default_op_Transpose(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Transpose_generic<uint8_t>,
			.int8_ = Transpose_generic<int8_t>,
			.int16_ = Transpose_generic<int16_t>,
			.int32_ = Transpose_generic<int32_t>,
			.int64_ = Transpose_generic<int64_t>,
			.uint8_ = Transpose_generic<uint8_t>,
			.uint16_ = Transpose_generic<uint16_t>,
			.uint32_ = Transpose_generic<uint32_t>,
			.uint64_ = Transpose_generic<uint64_t>,
			.bfloat16_ = Transpose_generic<uint16_t>,
			.float16_ = Transpose_generic<uint16_t>,
			.float32_ = Transpose_generic<float>,
			.float64_ = Transpose_generic<double>,
			.complex64_ = Transpose_generic<std::complex<float>>,
			.complex128_ = Transpose_generic<std::complex<double>>,
			.string_ = Transpose_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Transpose_generic<uint8_t>,
			.int8_ = Transpose_generic<int8_t>,
			.int16_ = Transpose_generic<int16_t>,
			.int32_ = Transpose_generic<int32_t>,
			.int64_ = Transpose_generic<int64_t>,
			.uint8_ = Transpose_generic<uint8_t>,
			.uint16_ = Transpose_generic<uint16_t>,
			.uint32_ = Transpose_generic<uint32_t>,
			.uint64_ = Transpose_generic<uint64_t>,
			.float16_ = Transpose_generic<uint16_t>,
			.float32_ = Transpose_generic<float>,
			.float64_ = Transpose_generic<double>,
			.complex64_ = Transpose_generic<std::complex<float>>,
			.complex128_ = Transpose_generic<std::complex<double>>,
			.string_ = Transpose_string,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Transpose_init;
		n->exit = Transpose_exit;
		n->reshape = Transpose_reshape;
	}
}
