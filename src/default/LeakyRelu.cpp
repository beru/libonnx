#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct LeakyRelu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 0.01f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] < 0) {
				py[i] = px[i] * alpha;
			}else {
				py[i] = px[i];
			}
		}
	}

	void exec() override {
		if (opset >= 6) {
			TYPED_EXEC(inputs[0]->type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_LeakyRelu()
{
	return new LeakyRelu_operator;
}

} // namespace onnx
