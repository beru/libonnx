#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Xor_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}

	bool exec_impl() {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			const bool_t* pa = (const bool_t*)a->broadcast_map_address(y, i);
			const bool_t* pb = (const bool_t*)b->broadcast_map_address(y, i);
			py[i] = (*pa != *pb);
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 7) {
			switch (type) {
			case ONNX_TENSOR_TYPE_BOOL:
				return exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 1) {
		}
		return false;
	}
};

} // namespace {

operator_t* resolver_default_op_Xor(int opset)
{
	return new (std::nothrow) Xor_operator;
}

} // namespace onnx
