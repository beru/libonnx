#include <onnx.h>
#include "float16.h"

namespace {

struct ope_pdata_t {
	float epsilon;
	float momentum;
};

int BatchNormalization_init(onnx_node_t* n)
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

int BatchNormalization_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int BatchNormalization_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
void BatchNormalization_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* mean = n->inputs[3];
	onnx_tensor_t* var = n->inputs[4];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* pscale = (T*)scale->datas;
	T* pb = (T*)b->datas;
	T* pmean = (T*)mean->datas;
	T* pvar = (T*)var->datas;
	T* py = (T*)y->datas;
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
		double denom1 = sqrt((double)pvar[jc] + pdat->epsilon);
		double denom2 = pb[jc];
		for (i = 0; i < channel; i++) {
			py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / denom1) + denom2;
		}
	}
}

} // namespace

void resolver_default_op_BatchNormalization(onnx_node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = BatchNormalization_generic<float16_t>,
			.float32_ = BatchNormalization_generic<float>,
			.float64_ = BatchNormalization_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = BatchNormalization_generic<float16_t>,
			.float32_ = BatchNormalization_generic<float>,
			.float64_ = BatchNormalization_generic<double>,
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
