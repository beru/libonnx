#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Shrink_operator : public operator_t {
	float bias;
	float lambd;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		bias = attribute("bias", 0.0f);
		lambd = attribute("lambd", 0.5f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] < -lambd)
				py[i] = px[i] + (T)bias;
			else if (px[i] > lambd)
				py[i] = px[i] - (T)bias;
			else
				py[i] = 0;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 9) {
			TYPED_EXEC(type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Shrink(int opset)
{
	return new Shrink_operator;
}

} // namespace onnx
