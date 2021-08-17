#include <onnx.h>

static int Erf_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Erf_exit(onnx_node_t* n)
{
	return 1;
}

static int Erf_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Erf_int8(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->datas;
	int16_t* py = (int16_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_int16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->datas;
	int16_t* py = (int16_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_int32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int32_t* px = (int32_t*)x->datas;
	int32_t* py = (int32_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_int64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t* px = (int64_t*)x->datas;
	int64_t* py = (int64_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erf(px[i]);
}

static void Erf_uint8(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_uint16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_uint32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint32_t* px = (uint32_t*)x->datas;
	uint32_t* py = (uint32_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_uint64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint64_t* px = (uint64_t*)x->datas;
	uint64_t* py = (uint64_t*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++)
		py[i] = erf(px[i]);
}

static void Erf_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(erff(v));
	}
}

static void Erf_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(erff(v));
	}
}

static void Erf_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = erff(px[i]);
}

static void Erf_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = erf(px[i]);
}

void resolver_default_op_Erf(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Erf_int8,
			.int16_ = Erf_int16,
			.int32_ = Erf_int32,
			.int64_ = Erf_int64,
			.uint8_ = Erf_uint8,
			.uint16_ = Erf_uint16,
			.uint32_ = Erf_uint32,
			.uint64_ = Erf_uint64,
			.bfloat16_ = Erf_bfloat16,
			.float16_ = Erf_float16,
			.float32_ = Erf_float32,
			.float64_ = Erf_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Erf_int8,
			.int16_ = Erf_int16,
			.int32_ = Erf_int32,
			.int64_ = Erf_int64,
			.uint8_ = Erf_uint8,
			.uint16_ = Erf_uint16,
			.uint32_ = Erf_uint32,
			.uint64_ = Erf_uint64,
			.float16_ = Erf_float16,
			.float32_ = Erf_float32,
			.float64_ = Erf_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Erf_init;
		n->exit = Erf_exit;
		n->reshape = Erf_reshape;
	}
}
