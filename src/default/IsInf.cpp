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

template <typename T>
static void IsInf_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
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
		n->ope = onnx_ope_type_selector{
			.float32_ = IsInf_generic<float>,
			.float64_ = IsInf_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = IsInf_init;
		n->exit = IsInf_exit;
		n->reshape = IsInf_reshape;
	}
}
