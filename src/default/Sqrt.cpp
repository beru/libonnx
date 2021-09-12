#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sqrt_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){ return sqrt(x); });
	}

	void exec() override {
		if (opset >= 13) {
			TYPED_EXEC(inputs[0]->type,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 6) {
			TYPED_EXEC(inputs[0]->type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Sqrt()
{
	return new Sqrt_operator;
}

} // namespace onnx
