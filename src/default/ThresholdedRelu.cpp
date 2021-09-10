#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct ThresholdedRelu_operator : public operator_t {
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
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = (px[i] > alpha) ? px[i] : (T)0;
	}
};

void resolver_default_op_ThresholdedRelu(node_t* n)
{
	if (n->opset >= 10) {
		n->ope = ope_type_select<ThresholdedRelu_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
