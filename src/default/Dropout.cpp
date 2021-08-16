#include <onnx.h>

static int Dropout_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() >= 1))
		return 1;
	return 0;
}

static int Dropout_exit(onnx_node_t* n)
{
	return 1;
}

static int Dropout_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Dropout_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

static void Dropout_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = px[i];
}

void resolver_default_op_Dropout(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Dropout_bfloat16,
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Dropout_float16,
			.float32_ = Dropout_float32,
			.float64_ = Dropout_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Dropout_init;
		n->exit = Dropout_exit;
		n->reshape = Dropout_reshape;
	}
}
