#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Reciprocal_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	};

	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return T(1.0)/x;});
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<Reciprocal_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<Reciprocal_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<Reciprocal_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Reciprocal(int opset) { return new (std::nothrow) Reciprocal_operator; }

} // namespace onnx
