#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	float alpha;
};

bool LeakyRelu_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->read_attribute("alpha", 0.01f);
	n->priv = pdat;
	return true;
}

template <typename T>
void LeakyRelu_generic(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		if (px[i] < 0) {
			py[i] = px[i] * pdat->alpha;
		}else {
			py[i] = px[i];
		}
}

GEN_HOLEDR_TYPE(holder, LeakyRelu_generic)

} // namespace

void resolver_default_op_LeakyRelu(node_t* n)
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
		n->init = LeakyRelu_init;
	}
}

} // namespace onnx
