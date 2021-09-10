#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Softplus_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){ return log(exp(x) + 1); });
	}

};

void resolver_default_op_Softplus(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = ope_type_select<Softplus_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
