#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Round_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){return rint(x);});
	}

};

void resolver_default_op_Round(node_t* n)
{
	if (n->opset >= 11) {
		n->ope = ope_type_select<Round_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
