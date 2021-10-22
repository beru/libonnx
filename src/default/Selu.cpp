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
		alpha = attribute("alpha", 1.67326319217681884765625f);
		gamma = attribute("gamma", 1.05070102214813232421875f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
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
			typed_exec<Selu_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Selu_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Selu(int opset)
{
	return new (std::nothrow) Selu_operator;
}

} // namespace onnx
