#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Relu_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return std::max((T)0, x);});
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 14) {
			TYPED_EXEC(type,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 13) {
			TYPED_EXEC(type,
				bfloat16_t, float16_t, float, double
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

operator_t* resolver_default_op_Relu(int opset)
{
	return new Relu_operator;
}

} // namespace onnx
