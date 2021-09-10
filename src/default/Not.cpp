#include <onnx.h>
#include "util.h"

namespace onnx {

struct Not_operator : public operator_t {

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
		const bool_t* px = (const bool_t*)x->data;
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = !px[i];
	}

};

void resolver_default_op_Not(node_t* n)
{
	if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BOOL:
			n->ope = std::make_shared<Not_operator>();
			break;
		default:
			break;
		}
	}
}

} // namespace onnx
