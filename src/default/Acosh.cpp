#include <onnx.h>
#include "util.h"

namespace onnx {

struct Acosh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return acosh(x);});
	}

	void exec() override {
		if (n->opset >= 9) {
			typed_exec<Acosh_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

void resolver_default_op_Acosh(node_t* n)
{
	n->ope = std::make_shared<Acosh_operator>();
}

} // namespace onnx
