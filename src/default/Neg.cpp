#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Neg_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return -x;});
	}

	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Neg_operator,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Neg_operator,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Neg_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}

};

} // namespace {

void resolver_default_op_Neg(node_t* n)
{
	n->ope = std::make_shared<Neg_operator>();
}

} // namespace onnx
