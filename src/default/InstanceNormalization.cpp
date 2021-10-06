#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct InstanceNormalization_operator : public operator_t {
	float epsilon;

	bool init() override {
		if (!(inputs.size() == 3 && outputs.size() >= 1)) {
			return false;
		}
		epsilon = attribute("epsilon", 1e-05f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		const tensor_t* scale = inputs[1];
		const tensor_t* b = inputs[2];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		const T* pscale = (const T*)scale->data;
		const T* pb = (const T*)b->data;
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
			int l = o + channel;
			int jc = j % C;
			T temp = 0;
			for (int i = o; i < l; i++) {
				temp += px[i];
			}
			T mean = temp / channel;
			temp = 0;
			for (int i = o; i < l; i++) {
				temp += pow(px[i] - mean, 2);
			}
			T var = temp / channel;
			double denom = sqrt((double)var + epsilon);
			double tmp = pb[jc];
			for (int i = o; i < l; i++) {
				py[i] = pscale[jc] * ((px[i] - mean) / denom) + tmp;
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

operator_t* resolver_default_op_InstanceNormalization(int opset)
{
	return new InstanceNormalization_operator;
}

} // namespace onnx
