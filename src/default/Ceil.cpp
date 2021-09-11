#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Ceil_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return ceil(x);});
	}
	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Ceil_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
			typed_exec<Ceil_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<Ceil_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_Ceil(node_t* n)
{
	n->ope = std::make_shared<Ceil_operator>();
}

} // namespace onnx
