#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Exp_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){return exp(x);});
	}

};

void resolver_default_op_Exp(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Exp_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Exp_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Exp_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
