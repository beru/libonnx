#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct BitShift_operator : public operator_t {
	bool isleft;

	bool init() override {
		if (!is_inout_size(2, 1)) {
			return false;
		}
		isleft = (strcmp(n->attribute("direction", "LEFT"), "LEFT") == 0);
		return true;
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		return y->reshape_multi_broadcast(a, b, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		T* py = (T*)y->data;

		if (isleft) {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				T* pa = (T*)a->broadcast_map_address(y, i);
				T* pb = (T*)b->broadcast_map_address(y, i);
				py[i] = *pa << *pb;
			}
		}else {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				T* pa = (T*)a->broadcast_map_address(y, i);
				T* pb = (T*)b->broadcast_map_address(y, i);
				py[i] = *pa >> *pb;
			}
		}
	}

	void exec() override {
		if (n->opset >= 11) {
			TYPED_EXEC(n->inputs[0]->type,
				uint8_t, uint16_t, uint32_t, uint64_t
			)
		}
	}

};

} // namespace {

void resolver_default_op_BitShift(node_t* n)
{
	n->ope = new BitShift_operator;
}

} // namespace onnx
