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
			typed_exec<Sqrt_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Sqrt_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Sqrt_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_Sqrt(node_t* n)
{
	n->ope = std::make_shared<Sqrt_operator>();
}

} // namespace onnx
