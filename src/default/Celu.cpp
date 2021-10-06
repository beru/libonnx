#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Celu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 1.0f);
		return true;
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 12) {
			switch (type) {
			case ONNX_TENSOR_TYPE_FLOAT32:
			{
				const tensor_t* x = inputs[0];
				tensor_t* y = outputs[0];
				const float* px = (const float*)x->data;
				float* py = (float*)y->data;
				for (size_t i = 0, l = y->ndata; i < l; i++) {
					py[i] = max((float)0.0, (float)px[i]) + min((float)0.0, (float)alpha * (expf(px[i] / alpha) - 1));
				}
			}
				break;
			default:
				break;
			}
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Celu(int opset)
{
	return new Celu_operator;
}

} // namespace onnx
