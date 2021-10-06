#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct BatchNormalization_operator : public operator_t {
	float epsilon;
	float momentum;

	bool init() override {
		if (!(inputs.size() == 5 && outputs.size() >= 1)) {
			return false;
		}
		epsilon = attribute("epsilon", 1e-05f);
		momentum = attribute("momentum", 0.9f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		const tensor_t* scale = inputs[1];
		const tensor_t* b = inputs[2];
		const tensor_t* mean = inputs[3];
		const tensor_t* var = inputs[4];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		const T* pscale = (const T*)scale->data;
		const T* pb = (const T*)b->data;
		const T* pmean = (const T*)mean->data;
		const T* pvar = (const T*)var->data;
		T* py = (T*)y->data;
		int N = x->dims[0];
		int C = x->dims[1];
		int NC = N * C;
		int channel = 1;
		for (int i = 2; i < x->ndim; i++) {
			channel *= x->dims[i];
		}
		for (int j = 0; j < NC; j++) {
			int o = j * channel;
			int jc = j % C;
			double denom1 = sqrt((double)pvar[jc] + epsilon);
			double denom2 = pb[jc];
			for (int i = 0; i < channel; i++) {
				py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / denom1) + denom2;
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 14) {
			;
		}else if (opset >= 9) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 7) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 6) {
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_BatchNormalization(int opset)
{
	return new BatchNormalization_operator;
}

} // namespace onnx
