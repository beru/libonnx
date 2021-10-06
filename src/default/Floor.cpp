#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Floor_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return floor(x);});
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Floor_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Floor_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Floor_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Floor(int opset)
{
	return new Floor_operator;
}

} // namespace onnx
