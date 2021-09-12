#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct IsNaN_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
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

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = isnan(px[i]) ? 1 : 0;
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 9) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_IsNaN(node_t* n)
{
	n->ope = new IsNaN_operator;
}

} // namespace onnx
