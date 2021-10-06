#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct BitShift_operator : public operator_t {
	bool isleft;

	bool init() override {
		if (!is_inout_size(2, 1)) {
			return false;
		}
		isleft = attribute("direction", "LEFT") == "LEFT";
		return true;
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		return y->reshape_multi_broadcast(a, b, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		T* py = (T*)y->data;

		if (isleft) {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				const T* pa = (const T*)a->broadcast_map_address(y, i);
				const T* pb = (const T*)b->broadcast_map_address(y, i);
				py[i] = *pa << *pb;
			}
		}else {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				const T* pa = (const T*)a->broadcast_map_address(y, i);
				const T* pb = (const T*)b->broadcast_map_address(y, i);
				py[i] = *pa >> *pb;
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 11) {
			typed_exec<BitShift_operator,
				uint8_t, uint16_t, uint32_t, uint64_t
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_BitShift(int opset)
{
	return new BitShift_operator;
}

} // namespace onnx
