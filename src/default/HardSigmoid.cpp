#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	float alpha;
	float beta;
};

bool HardSigmoid_init(node_t* n)
{
	if (!(n->inputs.size() > 0 && n->outputs.size() > 0)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->read_attribute("alpha", 0.2f);
	pdat->beta = n->read_attribute("beta", 0.5f);
	n->priv = pdat;
	return true;
}

template <typename T>
void HardSigmoid_generic(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = max((T)0.0, min((T)1.0, (T)(pdat->alpha * px[i] + pdat->beta)));
}

GEN_HOLEDR_TYPE(holder, HardSigmoid_generic)

} // namespace

void resolver_default_op_HardSigmoid(node_t* n)
{
	if (n->opset >= 6) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = HardSigmoid_init;
	}
}

} // namespace onnx
