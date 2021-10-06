#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Selu_operator : public operator_t {
	float alpha;
	float gamma;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 1.67326f);
		gamma = attribute("gamma", 1.0507f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] > 0) {
				py[i] = gamma * px[i];
			}else {
				py[i] = gamma * (alpha * exp(px[i]) - alpha);
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 6) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Selu(int opset)
{
	return new Selu_operator;
}

} // namespace onnx
