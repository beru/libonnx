#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Mean_operator : public operator_t {

	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		if (!y->reshape_identity(inputs[0])) {
			return false;
		}
		for (size_t i = 1; i < inputs.size(); ++i) {
			if (!y->reshape_multi_broadcast(y, inputs[i], y->type)) {
				return false;
			}
		}
		return true;
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			T sum = 0;
			for (size_t j = 0; j < inputs.size(); ++j) {
				const tensor_t* x = inputs[j];
				const T* px = (const T*)x->broadcast_map_address(y, i);
				sum += *px;
			}
			py[i] = sum / inputs.size();
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Mean_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 8) {
			typed_exec<Mean_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			typed_exec<Mean_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Mean_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Mean(int opset)
{
	return new Mean_operator;
}

} // namespace onnx
