#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Neg_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return -x;});
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<Neg_operator,
				int8_t, int16_t, int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<Neg_operator,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<Neg_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Neg(int opset) { return new (std::nothrow) Neg_operator; }

} // namespace onnx
