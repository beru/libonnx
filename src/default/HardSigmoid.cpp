#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct HardSigmoid_operator : public operator_t {
	float alpha;
	float beta;

	bool init() override {
		if (!(n->inputs.size() > 0 && n->outputs.size() > 0)) {
			return false;
		}
		alpha = n->attribute("alpha", 0.2f);
		beta = n->attribute("beta", 0.5f);
		return true;
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = max((T)0.0, min((T)1.0, (T)(alpha * px[i] + beta)));
	}
};

void resolver_default_op_HardSigmoid(node_t* n)
{
	if (n->opset >= 6) {
		n->ope = ope_type_select<HardSigmoid_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->opset >= 1) {
		n->ope = ope_type_select<HardSigmoid_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
