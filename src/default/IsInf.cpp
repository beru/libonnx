#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	int detect_negative;
	int detect_positive;
};

bool IsInf_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->detect_negative = n->read_attribute("detect_negative", 1);
	pdat->detect_positive = n->read_attribute("detect_positive", 1);
	n->priv = pdat;
	return true;
}

int IsInf_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
void IsInf_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	uint8_t* py = (uint8_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (isinf(px[i])) {
			if ((pdat->detect_negative && (px[i] < 0)) || (pdat->detect_positive && (px[i] > 0)))
				py[i] = 1;
			else
				py[i] = 0;
		}else {
			py[i] = 0;
		}
	}
}

GEN_HOLEDR_TYPE(holder, IsInf_generic)

} // namespace

void resolver_default_op_IsInf(onnx_node_t* n)
{
	if (n->opset >= 10) {
		n->ope = onnx_ope_type_select<holder,
			float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = IsInf_init;
		n->reshape = IsInf_reshape;
	}
}
