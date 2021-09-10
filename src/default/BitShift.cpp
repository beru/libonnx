#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct BitShift_operator : public operator_t {
	bool isleft;

	bool init() override {
		if (!is_inout_size(n, 2, 1)) {
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

	void exec() override {
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
};

void resolver_default_op_BitShift(node_t* n)
{
	if (n->opset >= 11) {
		n->ope = ope_type_select<BitShift_operator,
			uint8_t, uint16_t, uint32_t, uint64_t
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
