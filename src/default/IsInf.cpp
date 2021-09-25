#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct IsInf_operator : public operator_t {
	int detect_negative;
	int detect_positive;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		detect_negative = attribute("detect_negative", 1);
		detect_positive = attribute("detect_positive", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		uint8_t* py = (uint8_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (isinf(px[i])) {
				if ((detect_negative && (px[i] < 0)) || (detect_positive && (px[i] > 0)))
					py[i] = 1;
				else
					py[i] = 0;
			}else {
				py[i] = 0;
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 10) {
			TYPED_EXEC(type,
				float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_IsInf()
{
	return new IsInf_operator;
}

} // namespace onnx
