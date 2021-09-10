#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Floor_operator : public operator_t {

	bool init() override {
		return is_inout_size(n, 1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){return floor(x);});
	}

};

void resolver_default_op_Floor(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Floor_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Floor_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Floor_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
