#include <onnx.h>

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

static void Elu_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16((px[i] < 0) ? (expf(v) - 1) * pdat->alpha : v);
	}
}

static void Elu_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (expf(px[i]) - 1) * pdat->alpha : px[i];
}

static void Elu_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (exp(px[i]) - 1) * pdat->alpha : px[i];
}

void resolver_default_op_Elu(onnx_node_t* n)
{
	if (n->opset >= 6) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Elu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Elu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Elu_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Elu_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Elu_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Elu_float64;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Elu_init;
		n->exit = Elu_exit;
		n->reshape = Elu_reshape;
	}
}
