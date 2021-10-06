#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Round_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return rint(x);});
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 11) {
			typed_exec<Round_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Round(int opset)
{
	return new Round_operator;
}

} // namespace onnx
