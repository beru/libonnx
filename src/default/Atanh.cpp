#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Atanh_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	bool exec() {
		foreach_tensor<T>([](auto x){return atanh(x);});
		return true;
	}
	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 9) {
			return typed_exec<Atanh_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Atanh(int opset) { return new (std::nothrow) Atanh_operator; }

} // namespace onnx
