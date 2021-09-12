#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sqrt_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){ return sqrt(x); });
	}

	void exec() override {
		if (n->opset >= 13) {
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

void resolver_default_op_Sqrt(node_t* n)
{
	n->ope = new Sqrt_operator;
}

} // namespace onnx
