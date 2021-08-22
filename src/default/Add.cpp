#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

namespace {

int Add_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Add_exit(onnx_node_t* n)
{
	return 1;
}

int Add_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
void Add_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (T*)a->broadcast_map_address(y, i);
		pb = (T*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}


void Add_13_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) + bfloat16_to_float32(*pb));
	}
}

} // namespace {

void resolver_default_op_Add(onnx_node_t* n)
{
	if (n->opset >= 14) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Add_generic<int8_t>,
			.int16_ = Add_generic<int16_t>,
			.int32_ = Add_generic<int32_t>,
			.int64_ = Add_generic<int64_t>,
			.uint8_ = Add_generic<uint8_t>,
			.uint16_ = Add_generic<uint16_t>,
			.uint32_ = Add_generic<uint32_t>,
			.uint64_ = Add_generic<uint64_t>,
			.bfloat16_ = Add_generic<bfloat16_t>,
			.float16_ = Add_generic<float16_t>,
			.float32_ = Add_generic<float>,
			.float64_ = Add_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Add_generic<int32_t>,
			.int64_ = Add_generic<int64_t>,
			.uint32_ = Add_generic<uint32_t>,
			.uint64_ = Add_generic<uint64_t>,
			.bfloat16_ = Add_13_bfloat16,
			.float16_ = Add_generic<float16_t>,
			.float32_ = Add_generic<float>,
			.float64_ = Add_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7)	{
		n->ope = onnx_ope_type_selector{
			.int32_ = Add_generic<int32_t>,
			.int64_ = Add_generic<int64_t>,
			.uint32_ = Add_generic<uint32_t>,
			.uint64_ = Add_generic<uint64_t>,
			.float16_ = Add_generic<float16_t>,
			.float32_ = Add_generic<float>,
			.float64_ = Add_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6)	{
	}else if (n->opset >= 1)	{
	}
	if (n->ope) {
		n->init = Add_init;
		n->exit = Add_exit;
		n->reshape = Add_reshape;
	}
}
