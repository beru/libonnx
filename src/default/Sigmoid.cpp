#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sigmoid_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){
			if (x >= 0)
				return (T)1.0 / ((T)1.0 + (T)exp(-1 * x));
			else
				return exp(x) / ((T)1.0 + (T)exp(x));
		});
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				bfloat16_t, float16_t, float, double
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

void resolver_default_op_Sigmoid(node_t* n)
{
	n->ope = std::make_shared<Sigmoid_operator>();
}

} // namespace onnx
