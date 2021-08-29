#include <onnx.h>
#include "util.h"

namespace {

bool Dropout_init(onnx_node_t* n)
{
	return (n->inputs.size() >= 1) && (n->outputs.size() >= 1);
}

template <typename T>
void Dropout_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return x;});
}

GEN_HOLEDR_TYPE(holder, Dropout_generic)

} // namespace

void resolver_default_op_Dropout(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Dropout_init;
	}
}
