#include <onnx.h>
#include "util.h"

namespace {

bool Expand_init(onnx_node_t* n)
{
	return is_inout_size(n, 2, 1);
}

int Expand_exit(onnx_node_t* n)
{
	return 1;
}

int Expand_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* s = n->inputs[1];
	int64_t* ps = (int64_t*)s->data;
	int ndim = max(x->ndim, (int)s->ndata);
	std::vector<int> dims(ndim);

	for (int i = x->ndim - 1, j = s->ndata - 1, k = ndim - 1; k >= 0; k--) {
		if (i < 0)
			dims[k] = ps[j--];
		else if (j < 0)
			dims[k] = x->dims[i--];
		else {
			if (x->dims[i] == ps[j])
				dims[k] = x->dims[i];
			else if ((x->dims[i] == 1) || (ps[j] == 1))
				dims[k] = (x->dims[i] > ps[j]) ? x->dims[i] : ps[j];
			else
				return 0;
			i--;
			j--;
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
void Expand_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	T* py = (T*)y->data;
	T* px = (T*)x->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (T*)x->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

void Expand_string(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	std::string* px = (std::string*)x->data;
	std::string* py = (std::string*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (std::string*)x->broadcast_map_address(y, i);
		py[i] = px[i];
	}
}

} // namespace

void resolver_default_op_Expand(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Expand_generic<uint8_t>,
			.int8_ = Expand_generic<int8_t>,
			.int16_ = Expand_generic<int16_t>,
			.int32_ = Expand_generic<int32_t>,
			.int64_ = Expand_generic<int64_t>,
			.uint8_ = Expand_generic<uint8_t>,
			.uint16_ = Expand_generic<uint16_t>,
			.uint32_ = Expand_generic<uint32_t>,
			.uint64_ = Expand_generic<uint64_t>,
			.bfloat16_ = Expand_generic<bfloat16_t>,
			.float16_ = Expand_generic<float16_t>,
			.float32_ = Expand_generic<float>,
			.float64_ = Expand_generic<double>,
			.complex64_ = Expand_generic<std::complex<float>>,
			.complex128_ = Expand_generic<std::complex<double>>,
			.string_ = Expand_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Expand_generic<uint8_t>,
			.int8_ = Expand_generic<int8_t>,
			.int16_ = Expand_generic<int16_t>,
			.int32_ = Expand_generic<int32_t>,
			.int64_ = Expand_generic<int64_t>,
			.uint8_ = Expand_generic<uint8_t>,
			.uint16_ = Expand_generic<uint16_t>,
			.uint32_ = Expand_generic<uint32_t>,
			.uint64_ = Expand_generic<uint64_t>,
			.float16_ = Expand_generic<float16_t>,
			.float32_ = Expand_generic<float>,
			.float64_ = Expand_generic<double>,
			.complex64_ = Expand_generic<std::complex<float>>,
			.complex128_ = Expand_generic<std::complex<double>>,
			.string_ = Expand_string,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Expand_init;
		n->exit = Expand_exit;
		n->reshape = Expand_reshape;
	}
}
