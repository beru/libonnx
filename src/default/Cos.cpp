#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Cos_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return cos(x);});
	}
	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 7) {
			typed_exec<Cos_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Cos(int opset)
{
	return new Cos_operator;
}

} // namespace onnx
