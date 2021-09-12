#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Tanh_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){ return tanh(x); });
	}

	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Tanh_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Tanh_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Tanh_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_Tanh(node_t* n)
{
	n->ope = std::make_shared<Tanh_operator>();
}

} // namespace onnx
