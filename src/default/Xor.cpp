#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Xor_operator : public operator_t {

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
		uint8_t* py = (uint8_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			uint8_t* pa = (uint8_t*)a->broadcast_map_address(y, i);
			uint8_t* pb = (uint8_t*)b->broadcast_map_address(y, i);
			py[i] = (*pa != *pb) ? 1 : 0;
		}
	}

};

} // namespace {

void resolver_default_op_Xor(node_t* n)
{
	if (n->opset >= 7) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BOOL:
			n->ope = std::make_shared<Xor_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
