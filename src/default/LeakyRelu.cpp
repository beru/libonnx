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

static void LeakyRelu_float32(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

static void LeakyRelu_float64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? px[i] * pdat->alpha : px[i];
}

void resolver_default_op_LeakyRelu(onnx_node_t* n)
{
	if (n->opset >= 6) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = LeakyRelu_init;
			n->exit = LeakyRelu_exit;
			n->reshape = LeakyRelu_reshape;
			n->ope = LeakyRelu_float64;
			break;
		default:
			break;
		}
	}
}
