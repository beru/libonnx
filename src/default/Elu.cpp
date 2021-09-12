#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Elu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = n->attribute("alpha", 1.0f);
		return true;
	}

	template <typename T>
	void exec() {
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

void resolver_default_op_Elu(node_t* n)
{
	n->ope = std::make_shared<Elu_operator>();
}

} // namespace onnx
