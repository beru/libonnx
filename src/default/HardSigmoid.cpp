#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	float alpha;
	float beta;
};

bool HardSigmoid_init(onnx_node_t* n)
{
	if (!(n->inputs.size() > 0 && n->outputs.size() > 0)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->attribute_read_float("alpha", 0.2);
	pdat->beta = n->attribute_read_float("beta", 0.5);
	n->priv = pdat;
	return true;
}

int HardSigmoid_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int HardSigmoid_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void HardSigmoid_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = max((T)0.0, min((T)1.0, (T)(pdat->alpha * px[i] + pdat->beta)));
}

GEN_HOLEDR_TYPE(holder, HardSigmoid_generic)

} // namespace

void resolver_default_op_HardSigmoid(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = HardSigmoid_init;
		n->exit = HardSigmoid_exit;
		n->reshape = HardSigmoid_reshape;
	}
}
