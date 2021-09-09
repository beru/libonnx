#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float alpha;
};

bool Elu_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	if (!pdat)
		return false;
	pdat->alpha = n->attribute("alpha", 1.0f);
	n->priv = pdat;
	return true;
}

template <typename T>
void Elu_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		if (px[i] < 0) {
			py[i] = (exp(px[i]) - 1) * pdat->alpha;
		}else {
			py[i] = px[i];
		}
}

GEN_HOLEDR_TYPE(holder, Elu_generic)

} // namespace

void resolver_default_op_Elu(node_t* n)
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
		n->init = Elu_init;
	}
}

} // namespace onnx
