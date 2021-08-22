#include <onnx.h>

#include "float16.h"
#include "bfloat16.h"

namespace {

int Abs_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Abs_exit(onnx_node_t* n)
{
	return 1;
}

int Abs_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
void Abs_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if constexpr (std::is_signed_v<T>) {
			py[i] = abs(px[i]);
		}else {
			py[i] = px[i];
		}
	}
}

} // namespace {

void resolver_default_op_Abs(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Abs_generic<int8_t>,
			.int16_ = Abs_generic<int16_t>,
			.int32_ = Abs_generic<int32_t>,
			.int64_ = Abs_generic<int64_t>,
			.uint8_ = Abs_generic<uint8_t>,
			.uint16_ = Abs_generic<uint16_t>,
			.uint32_ = Abs_generic<uint32_t>,
			.uint64_ = Abs_generic<uint64_t>,
			.bfloat16_ = Abs_generic<bfloat16_t>,
			.float16_ = Abs_generic<float16_t>,
			.float32_ = Abs_generic<float>,
			.float64_ = Abs_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Abs_generic<int8_t>,
			.int16_ = Abs_generic<int16_t>,
			.int32_ = Abs_generic<int32_t>,
			.int64_ = Abs_generic<int64_t>,
			.uint8_ = Abs_generic<uint8_t>,
			.uint16_ = Abs_generic<uint16_t>,
			.uint32_ = Abs_generic<uint32_t>,
			.uint64_ = Abs_generic<uint64_t>,
			.float16_ = Abs_generic<float16_t>,
			.float32_ = Abs_generic<float>,
			.float64_ = Abs_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Abs_generic<float16_t>,
			.float32_ = Abs_generic<float>,
			.float64_ = Abs_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Abs_init;
		n->exit = Abs_exit;
		n->reshape = Abs_reshape;
	}
}
