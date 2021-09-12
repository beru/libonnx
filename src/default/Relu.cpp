#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Relu_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return (x < 0) ? (T)0 : x;});
	}

	void exec() override {
		if (n->opset >= 14) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_Relu(node_t* n)
{
	n->ope = new Relu_operator;
}

} // namespace onnx
