#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Softplus_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){ return log(exp(x) + 1); });
	}

	void exec() override {
		if (n->opset >= 1) {
			typed_exec<Softplus_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_Softplus(node_t* n)
{
	n->ope = std::make_shared<Softplus_operator>();
}

} // namespace onnx
