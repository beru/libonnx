#include <onnx.h>

struct operator_pdata_t {
	float alpha;
};

static int LeakyRelu_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 0.01);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int LeakyRelu_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int LeakyRelu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void LeakyRelu_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		if (v < 0)
			v *= pdat->alpha;
		py[i] = float32_to_float16(v);
	}
}

template <typename T>
static void LeakyRelu_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

void resolver_default_op_LeakyRelu(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = LeakyRelu_float16,
			.float32_ = LeakyRelu_generic<float>,
			.float64_ = LeakyRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = LeakyRelu_float16,
			.float32_ = LeakyRelu_generic<float>,
			.float64_ = LeakyRelu_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LeakyRelu_init;
		n->exit = LeakyRelu_exit;
		n->reshape = LeakyRelu_reshape;
	}
}
