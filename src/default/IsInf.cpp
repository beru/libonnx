#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct IsInf_operator : public operator_t {
	int detect_negative;
	int detect_positive;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		detect_negative = n->attribute("detect_negative", 1);
		detect_positive = n->attribute("detect_positive", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		uint8_t* py = (uint8_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (isinf(px[i])) {
				if ((detect_negative && (px[i] < 0)) || (detect_positive && (px[i] > 0)))
					py[i] = 1;
				else
					py[i] = 0;
			}else {
				py[i] = 0;
			}
		}
	}

	void exec() override {
		if (n->opset >= 10) {
			TYPED_EXEC(n->inputs[0]->type,
				float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_IsInf(node_t* n)
{
	n->ope = std::make_shared<IsInf_operator>();
}

} // namespace onnx
