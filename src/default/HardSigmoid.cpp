#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct HardSigmoid_operator : public operator_t {
	float alpha;
	float beta;

	bool init() override {
		if (!(inputs.size() > 0 && outputs.size() > 0)) {
			return false;
		}
		alpha = attribute("alpha", 0.2f);
		beta = attribute("beta", 0.5f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = max((T)0.0, min((T)1.0, (T)(alpha * px[i] + beta)));
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 6) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
		if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_HardSigmoid()
{
	return new HardSigmoid_operator;
}

} // namespace onnx
