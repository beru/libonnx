#include <onnx.h>

static int Round_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Round_exit(onnx_node_t* n)
{
	return 1;
}

static int Round_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Round_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(rintf(v));
	}
}

static void Round_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = rintf(px[i]);
}

static void Round_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = rint(px[i]);
}

void resolver_default_op_Round(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Round_float16,
			.float32_ = Round_float32,
			.float64_ = Round_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Round_init;
		n->exit = Round_exit;
		n->reshape = Round_reshape;
	}
}
