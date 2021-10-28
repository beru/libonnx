#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Exp_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return exp(x);});
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<Exp_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<Exp_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<Exp_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Exp(int opset) { return new (std::nothrow) Exp_operator; }

} // namespace onnx
