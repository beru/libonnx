#include <onnx.h>

struct operator_pdata_t {
	int detect_negative;
	int detect_positive;
};

static int IsInf_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->detect_negative = n->attribute_read_int("detect_negative", 1);
		pdat->detect_positive = n->attribute_read_int("detect_positive", 1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int IsInf_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int IsInf_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
}

static void IsInf_float32(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (isinf(px[i])) {
			if ((pdat->detect_negative && (px[i] < 0)) || (pdat->detect_positive && (px[i] > 0)))
				py[i] = 1;
			else
				py[i] = 0;
		}else {
			py[i] = 0;
		}
	}
}

static void IsInf_float64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (isinf(px[i])) {
			if ((pdat->detect_negative && (px[i] < 0)) || (pdat->detect_positive && (px[i] > 0)))
				py[i] = 1;
			else
				py[i] = 0;
		}else {
			py[i] = 0;
		}
	}
}

void resolver_default_op_IsInf(onnx_node_t* n)
{
	if (n->opset >= 10) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = IsInf_init;
			n->exit = IsInf_exit;
			n->reshape = IsInf_reshape;
			n->ope = IsInf_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = IsInf_init;
			n->exit = IsInf_exit;
			n->reshape = IsInf_reshape;
			n->ope = IsInf_float64;
			break;
		default:
			break;
		}
	}
}
