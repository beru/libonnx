#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Sum_operator : public operator_t {

	bool init() override {
		return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		int i;

		if (!y->reshape_identity(n->inputs[0]))
			return 0;
		for (i = 1; i < n->inputs.size(); i++) {
			if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
				return 0;
		}
		return 1;
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* x;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T sum = 0;
			for (size_t j = 0; j < n->inputs.size(); j++) {
				x = n->inputs[j];
				const T* px = (const T*)x->broadcast_map_address(y, i);
				sum += *px;
			}
			py[i] = sum;
		}
	}

};

void resolver_default_op_Sum(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Sum_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = ope_type_select<Sum_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Sum_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Sum_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
