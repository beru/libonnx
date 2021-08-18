#include <onnx.h>

struct ope_pdata_t {
	float bias;
	float lambd;
};

static int Shrink_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->bias = n->attribute_read_float("bias", 0.0);
		pdat->lambd = n->attribute_read_float("lambd", 0.5);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Shrink_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Shrink_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void Shrink_int8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->datas;
	int16_t* py = (int16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_int16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->datas;
	int16_t* py = (int16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_int32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int32_t* px = (int32_t*)x->datas;
	int32_t* py = (int32_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_int64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t* px = (int64_t*)x->datas;
	int64_t* py = (int64_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_uint8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_uint16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_uint32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint32_t* px = (uint32_t*)x->datas;
	uint32_t* py = (uint32_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_uint64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint64_t* px = (uint64_t*)x->datas;
	uint64_t* py = (uint64_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		if (v < -pdat->lambd)
			py[i] = float32_to_float16(v + pdat->bias);
		else if (v > pdat->lambd)
			py[i] = float32_to_float16(v - pdat->bias);
		else
			py[i] = float32_to_float16(0);
	}
}

static void Shrink_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

static void Shrink_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - pdat->bias;
		else
			py[i] = 0;
	}
}

void resolver_default_op_Shrink(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Shrink_int8,
			.int16_ = Shrink_int16,
			.int32_ = Shrink_int32,
			.int64_ = Shrink_int64,
			.uint8_ = Shrink_uint8,
			.uint16_ = Shrink_uint16,
			.uint32_ = Shrink_uint32,
			.uint64_ = Shrink_uint64,
			.float16_ = Shrink_float16,
			.float32_ = Shrink_float32,
			.float64_ = Shrink_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Shrink_init;
		n->exit = Shrink_exit;
		n->reshape = Shrink_reshape;
	}
}
