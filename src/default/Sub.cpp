#include <onnx.h>

static int Sub_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Sub_exit(onnx_node_t* n)
{
	return 1;
}

static int Sub_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
static void Sub_generic(onnx_node_t* n)
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
		py[i] = *pa - *pb;
	}
}

static void Sub_bfloat16(onnx_node_t* n)
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
		py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) - bfloat16_to_float32(*pb));
	}
}

static void Sub_float16(onnx_node_t* n)
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
		py[i] = float32_to_float16(float16_to_float32(*pa) - float16_to_float32(*pb));
	}
}

void resolver_default_op_Sub(onnx_node_t* n)
{
	if (n->opset >= 14) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Sub_generic<int8_t>,
			.int16_ = Sub_generic<int16_t>,
			.int32_ = Sub_generic<int32_t>,
			.int64_ = Sub_generic<int64_t>,
			.uint8_ = Sub_generic<uint8_t>,
			.uint16_ = Sub_generic<uint16_t>,
			.uint32_ = Sub_generic<uint32_t>,
			.uint64_ = Sub_generic<uint64_t>,
			.bfloat16_ = Sub_bfloat16,
			.float16_ = Sub_float16,
			.float32_ = Sub_generic<float>,
			.float64_ = Sub_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Sub_generic<int32_t>,
			.int64_ = Sub_generic<int64_t>,
			.uint32_ = Sub_generic<uint32_t>,
			.uint64_ = Sub_generic<uint64_t>,
			.bfloat16_ = Sub_bfloat16,
			.float16_ = Sub_float16,
			.float32_ = Sub_generic<float>,
			.float64_ = Sub_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Sub_generic<int32_t>,
			.int64_ = Sub_generic<int64_t>,
			.uint32_ = Sub_generic<uint32_t>,
			.uint64_ = Sub_generic<uint64_t>,
			.float16_ = Sub_float16,
			.float32_ = Sub_generic<float>,
			.float64_ = Sub_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Sub_init;
		n->exit = Sub_exit;
		n->reshape = Sub_reshape;
	}
}
