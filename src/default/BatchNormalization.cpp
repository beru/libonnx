#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct BatchNormalization_operator : public operator_t {
	float epsilon;
	float momentum;

	bool init() override {
		if (!(n->inputs.size() == 5 && n->outputs.size() >= 1)) {
			return false;
		}
		epsilon = n->attribute("epsilon", 1e-05f);
		momentum = n->attribute("momentum", 0.9f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		const tensor_t* scale = n->inputs[1];
		const tensor_t* b = n->inputs[2];
		const tensor_t* mean = n->inputs[3];
		const tensor_t* var = n->inputs[4];
		tensor_t* y = n->outputs[0];
		const T* px = (T*)x->data;
		const T* pscale = (T*)scale->data;
		const T* pb = (T*)b->data;
		const T* pmean = (T*)mean->data;
		const T* pvar = (T*)var->data;
		T* py = (T*)y->data;
		int N = x->dims[0];
		int C = x->dims[1];
		int NC = N * C;
		int channel = 1;
		int i, j, o, jc;

		for (i = 2; i < x->ndim; i++)
			channel *= x->dims[i];
		for (j = 0; j < NC; j++) {
			o = j * channel;
			jc = j % C;
			double denom1 = sqrt((double)pvar[jc] + epsilon);
			double denom2 = pb[jc];
			for (i = 0; i < channel; i++) {
				py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / denom1) + denom2;
			}
		}
	}

	void exec() override {
		if (n->opset >= 14) {
		}else if (n->opset >= 9) {
			typed_exec<BatchNormalization_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 7) {
			typed_exec<BatchNormalization_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 6) {
		}else if (n->opset >= 1) {
		}
	}

};

} // namespace {

void resolver_default_op_BatchNormalization(node_t* n)
{
	n->ope = std::make_shared<BatchNormalization_operator>();
}

} // namespace onnx
