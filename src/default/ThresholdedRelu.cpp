#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct ThresholdedRelu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 1.0f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			py[i] = (px[i] > alpha) ? px[i] : (T)0;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 10) {
			typed_exec<ThresholdedRelu_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_ThresholdedRelu(int opset)
{
	return new ThresholdedRelu_operator;
}

} // namespace onnx
