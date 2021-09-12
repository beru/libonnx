#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sin_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){ return sin(x); });
	}

	void exec() override {
		if (n->opset >= 7) {
			typed_exec<Sin_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_Sin(node_t* n)
{
	n->ope = std::make_shared<Sin_operator>();
}

} // namespace onnx
