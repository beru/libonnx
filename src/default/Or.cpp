#include <onnx.h>
#include "util.h"

namespace onnx {

struct Or_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			bool_t* pa = (bool_t*)a->broadcast_map_address(y, i);
			bool_t* pb = (bool_t*)b->broadcast_map_address(y, i);
			py[i] = (*pa || *pb);
		}
	}

};

void resolver_default_op_Or(node_t* n)
{
	if (n->opset >= 7) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BOOL:
			n->ope = std::make_shared<Or_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
