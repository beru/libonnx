#include <onnx.h>

static int Softsign_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Softsign_exit(onnx_node_t* n)
{
	return 1;
}

static int Softsign_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Softsign_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(v / (1 + fabsf(v)));
	}
}

static void Softsign_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i] / (1 + fabsf(px[i]));
}

static void Softsign_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i] / (1 + fabs(px[i]));
}

void resolver_default_op_Softsign(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Softsign_float16,
			.float32_ = Softsign_float32,
			.float64_ = Softsign_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Softsign_init;
		n->exit = Softsign_exit;
		n->reshape = Softsign_reshape;
	}
}
