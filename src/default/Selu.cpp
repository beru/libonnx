#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	float alpha;
	float gamma;
};

bool Selu_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->read_attribute("alpha", 1.67326f);
	pdat->gamma = n->read_attribute("gamma", 1.0507f);
	n->priv = pdat;
	return true;
}

template <typename T>
void Selu_generic(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] > 0)
			py[i] = pdat->gamma * px[i];
		else
			py[i] = pdat->gamma * (pdat->alpha * exp(px[i]) - pdat->alpha);
	}
}

GEN_HOLEDR_TYPE(holder, Selu_generic)

} // namespace

void resolver_default_op_Selu(node_t* n)
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
		n->init = Selu_init;
	}
}

} // namespace onnx
