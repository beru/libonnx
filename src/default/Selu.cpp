#include <onnx.h>
#include "float16.h"

struct operator_pdata_t {
	float alpha;
	float gamma;
};

static int Selu_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 1.67326);
		pdat->gamma = n->attribute_read_float("gamma", 1.0507);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Selu_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Selu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Selu_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] > 0)
			py[i] = pdat->gamma * px[i];
		else
			py[i] = pdat->gamma * (pdat->alpha * exp(px[i]) - pdat->alpha);
	}
}

void resolver_default_op_Selu(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Selu_generic<float16_t>,
			.float32_ = Selu_generic<float>,
			.float64_ = Selu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Selu_generic<float16_t>,
			.float32_ = Selu_generic<float>,
			.float64_ = Selu_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Selu_init;
		n->exit = Selu_exit;
		n->reshape = Selu_reshape;
	}
}
