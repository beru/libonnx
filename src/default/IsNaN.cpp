#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct IsNaN_operator : public operator_t {

	bool init() override {
		return is_inout_size(n, 1, 1);
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		uint8_t* py = (uint8_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = isnan(px[i]) ? 1 : 0;
	}

};

void resolver_default_op_IsNaN(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<IsNaN_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<IsNaN_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
