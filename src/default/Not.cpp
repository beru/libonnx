#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Not_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
	}

	void exec_impl() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const bool_t* px = (const bool_t*)x->data;
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = !px[i];
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 1) {
			switch (type) {
			case ONNX_TENSOR_TYPE_BOOL:
				exec_impl();
				break;
			default:
				break;
			}
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Not()
{
	return new Not_operator;
}

} // namespace onnx
