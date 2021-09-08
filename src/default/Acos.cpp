#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void Acos_generic(node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return acos(x);});
}

GEN_HOLEDR_TYPE(holder, Acos_generic)

} // namespace {

void resolver_default_op_Acos(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = [](node_t* n) {
			return is_inout_size(n, 1, 1);
		};
	}
}

} // namespace onnx
