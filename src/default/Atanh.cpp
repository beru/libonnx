#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Atanh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return atanh(x);});
	}
	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 9) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Atanh(int opset)
{
	return new Atanh_operator;
}

} // namespace onnx
