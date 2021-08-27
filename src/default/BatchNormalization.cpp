#include <onnx.h>
#include "util.h"

namespace {

struct ope_pdata_t : public onnx_node_t::ope_pdata_t {
	float epsilon;
	float momentum;
};

bool BatchNormalization_init(onnx_node_t* n)
{
	if (!(n->inputs.size() == 5 && n->outputs.size() >= 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->epsilon = n->attribute_read_float("epsilon", 1e-05);
	pdat->momentum = n->attribute_read_float("momentum", 0.9);
	n->priv = pdat;
	return true;
}

int BatchNormalization_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
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
	T* px = (T*)x->data;
	T* pscale = (T*)scale->data;
	T* pb = (T*)b->data;
	T* pmean = (T*)mean->data;
	T* pvar = (T*)var->data;
	T* py = (T*)y->data;
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

GEN_HOLEDR_TYPE(holder, BatchNormalization_generic)

} // namespace

void resolver_default_op_BatchNormalization(onnx_node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = BatchNormalization_init;
		n->reshape = BatchNormalization_reshape;
	}
}
