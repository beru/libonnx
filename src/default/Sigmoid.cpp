#include <onnx.h>
#include "util.h"

namespace {

template <typename T>
void Sigmoid_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){
		if (x >= 0)
			return (T)1.0 / ((T)1.0 + (T)exp(-1 * x));
		else
			return exp(x) / ((T)1.0 + (T)exp(x));
	});
}

GEN_HOLEDR_TYPE(holder, Sigmoid_generic)

} // namespace

void resolver_default_op_Sigmoid(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
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
		n->init = [](onnx_node_t* n){
			return is_inout_size(n, 1, 1);
		};
	}
}
