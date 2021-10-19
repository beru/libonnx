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
			typed_exec<Relu_operator,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 13) {
			typed_exec<Relu_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Relu_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Relu_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Relu(int opset)
{
	return new (std::nothrow) Relu_operator;
}

} // namespace onnx
