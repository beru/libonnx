#include <onnx.h>

struct ope_pdata_t {
	float epsilon;
	float momentum;
};

static int BatchNormalization_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 5) && (n->outputs.size() >= 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->epsilon = n->attribute_read_float("epsilon", 1e-05);
		pdat->momentum = n->attribute_read_float("momentum", 0.9);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int BatchNormalization_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int BatchNormalization_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void BatchNormalization_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* mean = n->inputs[3];
	onnx_tensor_t* var = n->inputs[4];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* pscale = (uint16_t*)scale->datas;
	uint16_t* pb = (uint16_t*)b->datas;
	uint16_t* pmean = (uint16_t*)mean->datas;
	uint16_t* pvar = (uint16_t*)var->datas;
	uint16_t* py = (uint16_t*)y->datas;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, o, jc;

	for (i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for (j = 0; j < NC; j++) {
		o = j * channel;
		jc = j % C;
		for (i = 0; i < channel; i++)
			py[o + i] = float32_to_float16(float16_to_float32(pscale[jc]) * ((float16_to_float32(px[o + i]) - float16_to_float32(pmean[jc])) / sqrtf(float16_to_float32(pvar[jc]) + pdat->epsilon)) + float16_to_float32(pb[jc]));
	}
}

static void BatchNormalization_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* mean = n->inputs[3];
	onnx_tensor_t* var = n->inputs[4];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* pscale = (float*)scale->datas;
	float* pb = (float*)b->datas;
	float* pmean = (float*)mean->datas;
	float* pvar = (float*)var->datas;
	float* py = (float*)y->datas;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, o, jc;

	for (i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for (j = 0; j < NC; j++) {
		o = j * channel;
		jc = j % C;
		for (i = 0; i < channel; i++)
			py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrtf(pvar[jc] + pdat->epsilon)) + pb[jc];
	}
}

static void BatchNormalization_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* mean = n->inputs[3];
	onnx_tensor_t* var = n->inputs[4];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* pscale = (double*)scale->datas;
	double* pb = (double*)b->datas;
	double* pmean = (double*)mean->datas;
	double* pvar = (double*)var->datas;
	double* py = (double*)y->datas;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, o, jc;

	for (i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for (j = 0; j < NC; j++) {
		o = j * channel;
		jc = j % C;
		for (i = 0; i < channel; i++)
			py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrt(pvar[jc] + pdat->epsilon)) + pb[jc];
	}
}

void resolver_default_op_BatchNormalization(onnx_node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = BatchNormalization_float16,
			.float32_ = BatchNormalization_float32,
			.float64_ = BatchNormalization_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = BatchNormalization_float16,
			.float32_ = BatchNormalization_float32,
			.float64_ = BatchNormalization_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = BatchNormalization_init;
		n->exit = BatchNormalization_exit;
		n->reshape = BatchNormalization_reshape;
	}
}
