#include <onnx.h>

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

void Add_int8(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int8_t* py = (int8_t*)y->datas;
	int8_t* pa;
	int8_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (int8_t*)a->broadcast_map_address(y, i);
		pb = (int8_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_int16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int16_t* py = (int16_t*)y->datas;
	int16_t* pa;
	int16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (int16_t*)a->broadcast_map_address(y, i);
		pb = (int16_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_int32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int32_t* py = (int32_t*)y->datas;
	int32_t* pa;
	int32_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (int32_t*)a->broadcast_map_address(y, i);
		pb = (int32_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_int64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int64_t* py = (int64_t*)y->datas;
	int64_t* pa;
	int64_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (int64_t*)a->broadcast_map_address(y, i);
		pb = (int64_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_uint8(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	uint8_t* pa;
	uint8_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint8_t*)a->broadcast_map_address(y, i);
		pb = (uint8_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_uint16(onnx_node_t* n)
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
		py[i] = *pa + *pb;
	}
}

void Add_uint32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint32_t* py = (uint32_t*)y->datas;
	uint32_t* pa;
	uint32_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint32_t*)a->broadcast_map_address(y, i);
		pb = (uint32_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_uint64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint64_t* py = (uint64_t*)y->datas;
	uint64_t* pa;
	uint64_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint64_t*)a->broadcast_map_address(y, i);
		pb = (uint64_t*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_float16(onnx_node_t* n)
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
		py[i] = float32_to_float16(float16_to_float32(*pa) + float16_to_float32(*pb));
	}
}

void Add_float32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	float* py = (float*)y->datas;
	float* pa;
	float* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (float*)a->broadcast_map_address(y, i);
		pb = (float*)b->broadcast_map_address(y, i);
		py[i] = *pa + *pb;
	}
}

void Add_float64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	double* py = (double*)y->datas;
	double* pa;
	double* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (double*)a->broadcast_map_address(y, i);
		pb = (double*)b->broadcast_map_address(y, i);
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
			.int8_ = Add_int8,
			.int16_ = Add_int16,
			.int32_ = Add_int32,
			.int64_ = Add_int64,
			.uint8_ = Add_uint8,
			.uint16_ = Add_uint16,
			.uint32_ = Add_uint32,
			.uint64_ = Add_uint64,
			.bfloat16_ = Add_13_bfloat16,
			.float16_ = Add_float16,
			.float32_ = Add_float32,
			.float64_ = Add_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Add_int32,
			.int64_ = Add_int64,
			.uint32_ = Add_uint32,
			.uint64_ = Add_uint64,
			.bfloat16_ = Add_13_bfloat16,
			.float16_ = Add_float16,
			.float32_ = Add_float32,
			.float64_ = Add_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7)	{
		n->ope = onnx_ope_type_selector{
			.int32_ = Add_int32,
			.int64_ = Add_int64,
			.uint32_ = Add_uint32,
			.uint64_ = Add_uint64,
			.float16_ = Add_float16,
			.float32_ = Add_float32,
			.float64_ = Add_float64,
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
