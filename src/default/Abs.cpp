#include <onnx.h>

static int Abs_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Abs_exit(onnx_node_t* n)
{
	return 1;
}

static int Abs_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Abs_generic(onnx_node_t* n)
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

static void Abs_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(fabsf(v));
	}
}

static void Abs_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(fabsf(v));
	}
}

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
			.bfloat16_ = Abs_bfloat16,
			.float16_ = Abs_float16,
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
			.float16_ = Abs_float16,
			.float32_ = Abs_generic<float>,
			.float64_ = Abs_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Abs_float16,
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
