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
	bool exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			T xv = px[i];
			py[i] = (xv > alpha) ? xv : (T)0;
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 10) {
			return typed_exec<ThresholdedRelu_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_ThresholdedRelu(int opset)
{
	return new (std::nothrow) ThresholdedRelu_operator;
}

} // namespace onnx
