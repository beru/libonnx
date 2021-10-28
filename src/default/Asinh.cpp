#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Asinh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return asinh(x);});
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 9) {
			typed_exec<Asinh_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Asinh(int opset) { return new (std::nothrow) Asinh_operator; }

} // namespace onnx
