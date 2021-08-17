#include <onnx.h>

static int Max_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Max_exit(onnx_node_t* n)
{
	return 1;
}

static int Max_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[0], n->inputs[0]->type))
		return 0;
	for (i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Max_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	T* py = (T*)y->datas;
	T* px;
	T maxv;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		maxv = std::numeric_limits<T>::min();
		for (int j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (T*)x->broadcast_map_address(y, i);
			if (*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float v;
	float maxv;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		maxv = FLT_MIN;
		for (int j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			v = bfloat16_to_float32(*px);
			if (v > maxv)
				maxv = v;
		}
		py[i] = float32_to_bfloat16(maxv);
	}
}

static void Max_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float v;
	float maxv;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		maxv = FLT_MIN;
		for (int j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			v = float16_to_float32(*px);
			if (v > maxv)
				maxv = v;
		}
		py[i] = float32_to_float16(maxv);
	}
}

void resolver_default_op_Max(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Max_generic<int8_t>,
			.int16_ = Max_generic<int16_t>,
			.int32_ = Max_generic<int32_t>,
			.int64_ = Max_generic<int64_t>,
			.uint8_ = Max_generic<uint8_t>,
			.uint16_ = Max_generic<uint16_t>,
			.uint32_ = Max_generic<uint32_t>,
			.uint64_ = Max_generic<uint64_t>,
			.bfloat16_ = Max_bfloat16,
			.float16_ = Max_float16,
			.float32_ = Max_generic<float>,
			.float64_ = Max_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Max_generic<int8_t>,
			.int16_ = Max_generic<int16_t>,
			.int32_ = Max_generic<int32_t>,
			.int64_ = Max_generic<int64_t>,
			.uint8_ = Max_generic<uint8_t>,
			.uint16_ = Max_generic<uint16_t>,
			.uint32_ = Max_generic<uint32_t>,
			.uint64_ = Max_generic<uint64_t>,
			.float16_ = Max_float16,
			.float32_ = Max_generic<float>,
			.float64_ = Max_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Max_float16,
			.float32_ = Max_generic<float>,
			.float64_ = Max_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Max_float16,
			.float32_ = Max_generic<float>,
			.float64_ = Max_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Max_float16,
			.float32_ = Max_generic<float>,
			.float64_ = Max_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Max_init;
		n->exit = Max_exit;
		n->reshape = Max_reshape;
	}
}
