#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Elu_operator : public operator_t {
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
			T v = px[i];
			if (v < 0) {
				v = (exp(v) - 1) * alpha;
			}
			py[i] = v;
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

operator_t* resolver_default_op_Elu()
{
	return new Elu_operator;
}

} // namespace onnx
