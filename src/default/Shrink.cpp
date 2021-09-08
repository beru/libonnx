#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float bias;
	float lambd;
};

bool Shrink_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->bias = n->read_attribute("bias", 0.0f);
	pdat->lambd = n->read_attribute("lambd", 0.5f);
	n->priv = pdat;
	return true;
}

template <typename T>
void Shrink_generic(node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + (T)pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - (T)pdat->bias;
		else
			py[i] = 0;
	}
}

GEN_HOLEDR_TYPE(holder, Shrink_generic)

} // namespace

void resolver_default_op_Shrink(node_t* n)
{
	if (n->opset >= 9) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Shrink_init;
	}
}

} // namespace onnx
