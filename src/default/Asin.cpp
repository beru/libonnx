#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Asin_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return asin(x);});
		return true;
	}
	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 7) {
			return typed_exec<Asin_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Asin(int opset) { return new (std::nothrow) Asin_operator; }

} // namespace onnx
