#include <onnx.h>
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
		alpha = n->attribute("alpha", 1.67326f);
		gamma = n->attribute("gamma", 1.0507f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] > 0)
				py[i] = gamma * px[i];
			else
				py[i] = gamma * (alpha * exp(px[i]) - alpha);
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

void resolver_default_op_Selu(node_t* n)
{
	n->ope = new Selu_operator;
}

} // namespace onnx
