#include <onnx.h>
#include "util.h"

namespace {

template <typename T>
void Acos_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return acos(x);});
}

GEN_HOLEDR_TYPE(holder, Acos_generic)

} // namespace {

void resolver_default_op_Acos(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = [](onnx_node_t* n) {
			return is_inout_size(n, 1, 1);
		};
	}
}

