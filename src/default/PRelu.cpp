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

template <typename T>
static void PRelu_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa = (T*)a->datas;;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			pb = (T*)b->broadcast_map_address(y, i);
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

void resolver_default_op_PRelu(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int32_ = PRelu_generic<int32_t>,
			.int64_ = PRelu_generic<int64_t>,
			.uint32_ = PRelu_generic<uint32_t>,
			.uint64_ = PRelu_generic<uint64_t>,
			.float16_ = PRelu_float16,
			.float32_ = PRelu_generic<float>,
			.float64_ = PRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_generic<float>,
			.float64_ = PRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_generic<float>,
			.float64_ = PRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = PRelu_float16,
			.float32_ = PRelu_generic<float>,
			.float64_ = PRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = PRelu_init;
		n->exit = PRelu_exit;
		n->reshape = PRelu_reshape;
	}
}
