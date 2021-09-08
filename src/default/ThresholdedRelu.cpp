#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float alpha;
};

bool ThresholdedRelu_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->read_attribute("alpha", 1.0f);
	n->priv = pdat;
	return true;
}

template <typename T>
void ThresholdedRelu_generic(node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] > pdat->alpha) ? px[i] : (T)0;
}

GEN_HOLEDR_TYPE(holder, ThresholdedRelu_generic)

} // namespace

void resolver_default_op_ThresholdedRelu(node_t* n)
{
	if (n->opset >= 10) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ThresholdedRelu_init;
	}
}

} // namespace onnx
