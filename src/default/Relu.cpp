#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Relu_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){return (x < 0) ? 0 : x;});
	}

};

void resolver_default_op_Relu(node_t* n)
{
	if (n->opset >= 14) {
		n->ope = ope_type_select<Relu_operator,
			int8_t, int16_t, int32_t, int64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = ope_type_select<Relu_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Relu_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Relu_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
