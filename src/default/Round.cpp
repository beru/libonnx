#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Round_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	bool exec() {
		int r = fegetround();
		if (r != FE_TONEAREST) {
			fesetround(FE_TONEAREST);
		}
		foreach_tensor<T>([](auto x){return nearbyint(x);});
		if (r != FE_TONEAREST) {
			fesetround(r);
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 11) {
			return typed_exec<Round_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Round(int opset) { return new (std::nothrow) Round_operator; }

} // namespace onnx
