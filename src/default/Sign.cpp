#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void Sign_generic(node_t* n)
{
	foreach_tensor<T>(n, [](auto x){
		if (x > 0)
			return 1;
		else if (x < 0)
			return -1;
		else
			return 0;
	});
}

GEN_HOLEDR_TYPE(holder, Sign_generic)

} // namespace

void resolver_default_op_Sign(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
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
