#include <onnx.h>
#include "util.h"

namespace onnx {

struct Celu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = n->attribute("alpha", 1.0f);
		return true;
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const float* px = (const float*)x->data;
		float* py = (float*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = max((float)0.0, (float)px[i]) + min((float)0.0, (float)alpha * (expf(px[i] / alpha) - 1));
	}
};

void resolver_default_op_Celu(node_t* n)
{
	if (n->opset >= 12) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = std::make_shared<Celu_operator>();
			break;
		default:
			break;
		}
	}
}

} // namespace onnx
