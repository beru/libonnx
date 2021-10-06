#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Abs_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if constexpr (std::is_signed_v<T>) {
				py[i] = abs(px[i]);
			}else {
				py[i] = px[i];
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Abs_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Abs_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Abs_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Abs(int opset)
{
	return new Abs_operator;
}

} // namespace onnx
