#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct InstanceNormalization_operator : public operator_t {
	float epsilon;

	bool init() override {
		if (!(n->inputs.size() == 3 && n->outputs.size() >= 1)) {
			return false;
		}
		epsilon = n->attribute("epsilon", 1e-05f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		const tensor_t* scale = n->inputs[1];
		const tensor_t* b = n->inputs[2];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		const T* pscale = (const T*)scale->data;
		const T* pb = (const T*)b->data;
		T* py = (T*)y->data;
		T temp, mean, var;
		int N = x->dims[0];
		int C = x->dims[1];
		int NC = N * C;
		int channel = 1;
		int i, j, l, o, jc;

		for (i = 2; i < x->ndim; i++)
			channel *= x->dims[i];
		for (j = 0; j < NC; j++) {
			o = j * channel;
			l = o + channel;
			jc = j % C;
			temp = 0;
			for (i = o; i < l; i++)
				temp += px[i];
			mean = temp / channel;
			temp = 0;
			for (i = o; i < l; i++)
				temp += pow(px[i] - mean, 2);
			var = temp / channel;
			double denom = sqrt((double)var + epsilon);
			double tmp = pb[jc];
			for (i = o; i < l; i++) {
				py[i] = pscale[jc] * ((px[i] - mean) / denom) + tmp;
			}
		}
	}

	void exec() override {
		if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_InstanceNormalization(node_t* n)
{
	n->ope = std::make_shared<InstanceNormalization_operator>();
}

} // namespace onnx
