#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Acosh_operator : public operator_t {
	bool init() override {
		return is_inout_size(n, 1, 1);
	}
	void exec() override {
		foreach_tensor<T>(n, [](auto x){return acosh(x);});
	}
};

void resolver_default_op_Acosh(node_t* n)
{
	if (n->opset >= 9) {
		n->ope = ope_type_select<Acosh_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
