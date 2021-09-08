#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	void* pmin;
	void* pmax;
};

bool Clip_init(node_t* n)
{
	if (!(n->inputs.size() >= 1 && n->outputs.size() == 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->pmin = nullptr;
	pdat->pmax = nullptr;
	n->priv = pdat;
	return true;
}

int Clip_reshape(node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];

	pdat->pmin = nullptr;
	pdat->pmax = nullptr;
	for (int i = 1; i < min<size_t>(3, n->inputs.size()); i++) {
		if (n->inputs[i]->ndim == 0) {
			if (n->inputs[i]->name == "min")
				pdat->pmin = n->inputs[i]->data;
			else if (n->inputs[i]->name == "max")
				pdat->pmax = n->inputs[i]->data;
		}
	}
	return y->reshape_identity(x);
}

template <typename T>
void Clip_generic(node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	T minv = pdat->pmin ? *(T*)pdat->pmin : std::numeric_limits<T>::lowest();
	T maxv = pdat->pmax ? *(T*)pdat->pmax : std::numeric_limits<T>::max();

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

GEN_HOLEDR_TYPE(holder, Clip_generic)

} // namespace

void resolver_default_op_Clip(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Clip_init;
		n->reshape = Clip_reshape;
	}

}

} // namespace onnx
