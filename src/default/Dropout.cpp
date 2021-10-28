#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Dropout_operator : public operator_t {
	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() >= 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return x;});
	}
	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Dropout_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 12) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 10) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Dropout_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Dropout(int opset) { return new (std::nothrow) Dropout_operator; }

} // namespace onnx
