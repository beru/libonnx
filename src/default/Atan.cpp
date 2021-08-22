#include <onnx.h>

namespace {

int Atan_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Atan_exit(onnx_node_t* n)
{
	return 1;
}

int Atan_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

void Atan_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		float v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(atanf(v));
	}
}

template <typename T>
void Atan_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atan(px[i]);
}

} // namespace

void resolver_default_op_Atan(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Atan_float16,
			.float32_ = Atan_generic<float>,
			.float64_ = Atan_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Atan_init;
		n->exit = Atan_exit;
		n->reshape = Atan_reshape;
	}
}
