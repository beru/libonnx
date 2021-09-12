#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Erf_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	template <typename T>
	void exec() {
		foreach_tensor<T>([](auto x){return erf(x);});
	}
	void exec() override {
		if (opset >= 13) {
			TYPED_EXEC(inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 9) {
			TYPED_EXEC(inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Erf()
{
	return new Erf_operator;
}

} // namespace onnx
