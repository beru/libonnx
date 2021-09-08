#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void Relu_generic(node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return (x < 0) ? 0 : x;});
}

GEN_HOLEDR_TYPE(holder, Relu_generic)

} // namespace

void resolver_default_op_Relu(node_t* n)
{
	if (n->opset >= 14) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = [](node_t* n){
			return is_inout_size(n, 1, 1);
		};
	}
}

} // namespace onnx
