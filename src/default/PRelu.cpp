#include <onnx.h>

static int PRelu_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int PRelu_exit(onnx_node_t* n)
{
	return 1;
}

static int PRelu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];

	return y->reshape_identity(a, a->type);
}

static void PRelu_int32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int32_t* py = (int32_t*)y->datas;
	int32_t* pa = (int32_t*)a->datas;;
	int32_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (int32_t*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_int64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int64_t* py = (int64_t*)y->datas;
	int64_t* pa = (int64_t*)a->datas;;
	int64_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (int64_t*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_uint32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint32_t* py = (uint32_t*)y->datas;
	uint32_t* pa = (uint32_t*)a->datas;;
	uint32_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (uint32_t*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_uint64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint64_t* py = (uint64_t*)y->datas;
	uint64_t* pa = (uint64_t*)a->datas;;
	uint64_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (uint64_t*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa = (uint16_t*)a->datas;;
	uint16_t* pb;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(pa[i]);
		if (v < 0) {
			pb = (uint16_t*)b->broadcast_map_address(y, i);
			py[i] = float32_to_float16(v * float16_to_float32(*pb));
		}
		else
			py[i] = float32_to_float16(v);
	}
}

static void PRelu_float32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	float* py = (float*)y->datas;
	float* pa = (float*)a->datas;;
	float* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (float*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_float64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	double* py = (double*)y->datas;
	double* pa = (double*)a->datas;;
	double* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (double*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

void resolver_default_op_PRelu(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int32_ = PRelu_int32,
			.int64_ = PRelu_int64,
			.uint32_ = PRelu_uint32,
			.uint64_ = PRelu_uint64,
			.float16_ = PRelu_float16,
			.float32_ = PRelu_float32,
			.float64_ = PRelu_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_float32,
			.float64_ = PRelu_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_float32,
			.float64_ = PRelu_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_float32,
			.float64_ = PRelu_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = PRelu_init;
		n->exit = PRelu_exit;
		n->reshape = PRelu_reshape;
	}
}
