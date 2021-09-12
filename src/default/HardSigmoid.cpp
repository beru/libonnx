#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

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

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = max((T)0.0, min((T)1.0, (T)(alpha * px[i] + beta)));
	}

	void exec() override {
		if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
		if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_HardSigmoid(node_t* n)
{
	n->ope = std::make_shared<HardSigmoid_operator>();
}

} // namespace onnx
