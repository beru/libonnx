#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Reciprocal_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	};

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return T(1.0)/x;});
	}

	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Reciprocal_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Reciprocal_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Reciprocal_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}

};

} // namespace {

void resolver_default_op_Reciprocal(node_t* n)
{
	n->ope = std::make_shared<Reciprocal_operator>();
}

} // namespace onnx
