#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Softplus_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){ return log(exp(x) + 1); });
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 1) {
			typed_exec<Softplus_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Softplus(int opset) { return new (std::nothrow) Softplus_operator; }

} // namespace onnx
