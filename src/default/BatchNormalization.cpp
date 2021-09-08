#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float epsilon;
	float momentum;
};

bool BatchNormalization_init(node_t* n)
{
	if (!(n->inputs.size() == 5 && n->outputs.size() >= 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	if (!pdat)
		return false;
	pdat->epsilon = n->attribute("epsilon", 1e-05f);
	pdat->momentum = n->attribute("momentum", 0.9f);
	n->priv = pdat;
	return true;
}

template <typename T>
void BatchNormalization_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* scale = n->inputs[1];
	tensor_t* b = n->inputs[2];
	tensor_t* mean = n->inputs[3];
	tensor_t* var = n->inputs[4];
	tensor_t* y = n->outputs[0];
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

void resolver_default_op_BatchNormalization(node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = BatchNormalization_init;
	}
}

} // namespace onnx
