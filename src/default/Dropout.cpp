#include <onnx.h>
#include "util.h"

namespace onnx {

struct Dropout_operator : public operator_t {
	bool init() override {
		return (n->inputs.size() >= 1) && (n->outputs.size() >= 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return x;});
	}
	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Dropout_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 12) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 10) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 7) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

void resolver_default_op_Dropout(node_t* n)
{
	n->ope = std::make_shared<Dropout_operator>();
}

} // namespace onnx
