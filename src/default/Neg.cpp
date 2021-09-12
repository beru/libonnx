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
		foreach_tensor<T>([](auto x){return -x;});
	}

	void exec() override {
		if (opset >= 13) {
			TYPED_EXEC(inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 6) {
			TYPED_EXEC(inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
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

operator_t* resolver_default_op_Neg()
{
	return new Neg_operator;
}

} // namespace onnx
