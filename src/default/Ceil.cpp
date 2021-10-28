#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Ceil_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return ceil(x);});
		return true;
	}
	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<Ceil_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<Ceil_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<Ceil_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Ceil(int opset) { return new (std::nothrow) Ceil_operator; }

} // namespace onnx
