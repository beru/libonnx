#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Tan_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){ return tan(x); });
	}

};

void resolver_default_op_Tan(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<Tan_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
