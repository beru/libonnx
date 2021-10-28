#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Dropout_operator : public operator_t {
	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() >= 1);
	}
	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return x;});
		return true;
	}
	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<Dropout_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 12) {
			return typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 10) {
			return typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			return typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Dropout(int opset) { return new (std::nothrow) Dropout_operator; }

} // namespace onnx
