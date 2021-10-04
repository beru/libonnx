#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sum_operator : public operator_t {

	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		int i;

		if (!y->reshape_identity(inputs[0]))
			return false;
		for (i = 1; i < inputs.size(); i++) {
			if (!y->reshape_multi_broadcast(y, inputs[i], y->type))
				return false;
		}
		return 1;
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T sum = 0;
			for (size_t j = 0; j < inputs.size(); j++) {
				const tensor_t* x = inputs[j];
				const T* px = (const T*)x->broadcast_map_address(y, i);
				sum += *px;
			}
			py[i] = sum;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(type,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 8) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 6) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Sum()
{
	return new Sum_operator;
}

} // namespace onnx
