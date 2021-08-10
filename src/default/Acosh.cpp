#include <onnx.h>

static int Acosh_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Acosh_exit(onnx_node_t* n)
{
	return 1;
}

static int Acosh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Acosh_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;
	size_t i, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(acoshf(v));
	}
}

static void Acosh_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;
	size_t i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = acoshf(px[i]);
}

static void Acosh_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;
	size_t i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = acosh(px[i]);
}

void resolver_default_op_Acosh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Acosh_float16,
			.float32_ = Acosh_float32,
			.float64_ = Acosh_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Acosh_init;
		n->exit = Acosh_exit;
		n->reshape = Acosh_reshape;
	}
}
