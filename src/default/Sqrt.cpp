#include "onnx.h"
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
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Sqrt_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Sqrt_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Sqrt_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Sqrt(int opset)
{
	return new Sqrt_operator;
}

} // namespace onnx
