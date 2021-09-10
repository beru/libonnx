#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Elu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(n, 1, 1)) {
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
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] < 0) {
				py[i] = (exp(px[i]) - 1) * alpha;
			}else {
				py[i] = px[i];
			}
		}
	}

};

void resolver_default_op_Elu(node_t* n)
{
	if (n->opset >= 6) {
		n->ope = ope_type_select<Elu_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Elu_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
