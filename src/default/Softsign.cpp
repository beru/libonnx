#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Softsign_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){ return x / (1 + fabs(x)); });
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Softsign(int opset)
{
	return new Softsign_operator;
}

} // namespace onnx
