#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Cos_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){return cos(x);});
	}
	void exec() override {
		if (n->opset >= 7) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_Cos(node_t* n)
{
	n->ope = new Cos_operator;
}

} // namespace onnx
