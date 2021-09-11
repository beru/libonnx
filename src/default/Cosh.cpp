#include <onnx.h>
#include "util.h"

namespace onnx {

struct Cosh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return cosh(x);});
	}
	void exec() override {
		if (n->opset >= 9) {
			typed_exec<Cosh_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

void resolver_default_op_Cosh(node_t* n)
{
	n->ope = std::make_shared<Cosh_operator>();
}

} // namespace onnx
