#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Atan_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	void exec() override {
		foreach_tensor<T>(n, [](auto x){return atan(x);});
	}
};

void resolver_default_op_Atan(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<Atan_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
