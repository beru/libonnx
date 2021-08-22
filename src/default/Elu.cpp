#include <onnx.h>
#include "float16.h"

struct ope_pdata_t {
	float alpha;
};

static int Elu_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 1.0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Elu_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Elu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Elu_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		if (px[i] < 0) {
			py[i] = (exp(px[i]) - 1) * pdat->alpha;
		}else {
			py[i] = px[i];
		}
}

void resolver_default_op_Elu(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Elu_generic<float16_t>,
			.float32_ = Elu_generic<float>,
			.float64_ = Elu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Elu_generic<float16_t>,
			.float32_ = Elu_generic<float>,
			.float64_ = Elu_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Elu_init;
		n->exit = Elu_exit;
		n->reshape = Elu_reshape;
	}
}
