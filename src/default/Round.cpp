#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Round_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return rint(x);});
	}

	void exec() override {
		if (n->opset >= 11) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_Round(node_t* n)
{
	n->ope = std::make_shared<Round_operator>();
}

} // namespace onnx
