#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Softsign_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){ return x / (1 + fabs(x)); });
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 1) {
			return typed_exec<Softsign_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Softsign(int opset) { return new (std::nothrow) Softsign_operator; }

} // namespace onnx
