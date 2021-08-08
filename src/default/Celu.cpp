#include <onnx.h>

struct ope_pdata_t {
	float alpha;
};

static int Celu_init(onnx_node_t* n)
{

	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 1.0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Celu_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Celu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Celu_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = max((float)0.0, (float)px[i]) + min((float)0.0, (float)pdat->alpha * (expf(px[i] / pdat->alpha) - 1));
}

void resolver_default_op_Celu(onnx_node_t* n)
{
	if (n->opset >= 12) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Celu_float32;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Celu_init;
		n->exit = Celu_exit;
		n->reshape = Celu_reshape;
	}
}
