#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Dropout_operator : public operator_t {
	bool init() override {
		return (n->inputs.size() >= 1) && (n->outputs.size() >= 1);
	}
	void exec() override {
		foreach_tensor<T>(n, [](auto x){return x;});
	}
};

void resolver_default_op_Dropout(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Dropout_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<Dropout_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = ope_type_select<Dropout_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<Dropout_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Dropout_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Dropout_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
