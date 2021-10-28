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
	bool exec() {
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
		int channel = multiply_accumulate(&x->dims[2], &x->dims[x->ndim], 1);
		for (int j = 0; j < NC; ++j) {
			int o = j * channel;
			int jc = j % C;
			double scalev = pscale[jc];
			double meanv = pmean[jc];
			double varv = pvar[jc];
			double denom = sqrt(varv + epsilon);
			double bv = pb[jc];
			for (int i = 0; i < channel; ++i) {
				py[o + i] = scalev * ((px[o + i] - meanv) / denom) + bv;
			}
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 14) {
		}else if (opset >= 9) {
			return typed_exec<BatchNormalization_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			return typed_exec<BatchNormalization_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
		}else if (opset >= 1) {
		}
		return false;
	}

};

} // namespace {

operator_t* resolver_default_op_BatchNormalization(int opset) { return new (std::nothrow) BatchNormalization_operator; }

} // namespace onnx
