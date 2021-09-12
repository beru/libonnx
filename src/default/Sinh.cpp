#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sinh_operator : public operator_t {
	
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){ return sinh(x); });
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

operator_t* resolver_default_op_Sinh()
{
	return new Sinh_operator;
}

} // namespace onnx
