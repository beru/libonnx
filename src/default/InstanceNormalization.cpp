#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	float epsilon;
};

bool InstanceNormalization_init(node_t* n)
{
	if (!(n->inputs.size() == 3 && n->outputs.size() >= 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	if (!pdat)
		return false;
	pdat->epsilon = n->attribute("epsilon", 1e-05f);
	n->priv = pdat;
	return true;
}

template <typename T>
void InstanceNormalization_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	const tensor_t* scale = n->inputs[1];
	const tensor_t* b = n->inputs[2];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
	const T* pscale = (const T*)scale->data;
	const T* pb = (const T*)b->data;
	T* py = (T*)y->data;
	T temp, mean, var;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, l, o, jc;

	for (i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for (j = 0; j < NC; j++) {
		o = j * channel;
		l = o + channel;
		jc = j % C;
		temp = 0;
		for (i = o; i < l; i++)
			temp += px[i];
		mean = temp / channel;
		temp = 0;
		for (i = o; i < l; i++)
			temp += pow(px[i] - mean, 2);
		var = temp / channel;
		double denom = sqrt((double)var + pdat->epsilon);
		double tmp = pb[jc];
		for (i = o; i < l; i++) {
			py[i] = pscale[jc] * ((px[i] - mean) / denom) + tmp;
		}
	}
}

GEN_HOLEDR_TYPE(holder, InstanceNormalization_generic)

} // namespace

void resolver_default_op_InstanceNormalization(node_t* n)
{
	if (n->opset >= 6) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = InstanceNormalization_init;
	}
}

} // namespace onnx
