#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
};

static int HardSigmoid_init(onnx_node_t* n)
{
	if ((n->inputs.size() > 0) && (n->outputs.size() > 0)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 0.2);
		pdat->beta = n->attribute_read_float("beta", 0.5);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int HardSigmoid_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int HardSigmoid_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void HardSigmoid_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(max((float)0.0, min((float)1.0, (float)(pdat->alpha * v + pdat->beta))));
	}
}

static void HardSigmoid_float32(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = max((float)0.0, min((float)1.0, (float)(pdat->alpha * px[i] + pdat->beta)));
}

static void HardSigmoid_float64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = max((double)0.0, min((double)1.0, (double)(pdat->alpha * px[i] + pdat->beta)));
}

void resolver_default_op_HardSigmoid(onnx_node_t* n)
{
	if (n->opset >= 6) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = HardSigmoid_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = HardSigmoid_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = HardSigmoid_float64;
			break;
		default:
			break;
		}
	}
	if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = HardSigmoid_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = HardSigmoid_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = HardSigmoid_float64;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = HardSigmoid_init;
		n->exit = HardSigmoid_exit;
		n->reshape = HardSigmoid_reshape;
	}
}
