#include <onnx.h>

namespace {

int Atanh_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Atanh_exit(onnx_node_t* n)
{
	return 1;
}

int Atanh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

void Atanh_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		float v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(atanhf(v));
	}
}

template <typename T>
void Atanh_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atanh(px[i]);
}

} // namespace

void resolver_default_op_Atanh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Atanh_float16,
			.float32_ = Atanh_generic<float>,
			.float64_ = Atanh_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Atanh_init;
		n->exit = Atanh_exit;
		n->reshape = Atanh_reshape;
	}
}
