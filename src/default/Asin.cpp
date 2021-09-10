#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Asin_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	void exec() override {
		foreach_tensor<T>(n, [](auto x){return asin(x);});
	}
};

void resolver_default_op_Asin(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<Asin_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
