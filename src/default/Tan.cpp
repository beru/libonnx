#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Tan_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){ return tan(x); });
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 7) {
			return typed_exec<Tan_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Tan(int opset)
{
	return new (std::nothrow) Tan_operator;
}

} // namespace onnx
