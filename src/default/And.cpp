#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct And_operator : public operator_t {
	bool init() override {
		return is_inout_size(2, 1);
	}
	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}
	void exec() override {
		if (opset >= 7) {
			switch (inputs[0]->type) {
			case ONNX_TENSOR_TYPE_BOOL:
			{
				tensor_t* y = outputs[0];
				const tensor_t* a = inputs[0];
				const tensor_t* b = inputs[1];
				bool_t* py = (bool_t*)y->data;
				for (size_t i = 0, l = y->ndata; i < l; i++) {
					const bool_t* pa = (const bool_t*)a->broadcast_map_address(y, i);
					const bool_t* pb = (const bool_t*)b->broadcast_map_address(y, i);
					py[i] = (*pa && *pb);
				}
				break;
			}
			default:
				break;
			}
		}else if (opset >= 1)	{
		}

	}
};

} // namespace {

operator_t* resolver_default_op_And()
{
	return new And_operator;
}

} // namespace onnx
