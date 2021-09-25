#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Dropout_operator : public operator_t {
	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() >= 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return x;});
	}
	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(type,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 12) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 10) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 7) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 6) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Dropout()
{
	return new Dropout_operator;
}

} // namespace onnx
