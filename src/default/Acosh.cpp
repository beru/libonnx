#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Acosh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return acosh(x);});
	}

	void exec() override {
		if (opset >= 9) {
			TYPED_EXEC(inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Acosh()
{
	return new Acosh_operator;
}

} // namespace onnx
