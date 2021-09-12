#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Softsign_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){ return x / (1 + fabs(x)); });
	}

	void exec() override {
		if (n->opset >= 1) {
			typed_exec<Softsign_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}

};

} // namespace {

void resolver_default_op_Softsign(node_t* n)
{
	n->ope = std::make_shared<Softsign_operator>();
}

} // namespace onnx
