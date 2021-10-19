#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Sigmoid_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){
			if (x >= 0) {
				return (T)1.0 / ((T)1.0 + (T)exp(-1 * x));
			}else {
				return exp(x) / ((T)1.0 + (T)exp(x));
			}
		});
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Sigmoid_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Sigmoid_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Sigmoid_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Sigmoid(int opset)
{
	return new (std::nothrow) Sigmoid_operator;
}

} // namespace onnx
