#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

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
			TYPED_EXEC(n->inputs[0]->type,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 12) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 10) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 7) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_Dropout(node_t* n)
{
	n->ope = std::make_shared<Dropout_operator>();
}

} // namespace onnx
