#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Sigmoid_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	void exec() override {
		foreach_tensor<T>(n, [](auto x){
			if (x >= 0)
				return (T)1.0 / ((T)1.0 + (T)exp(-1 * x));
			else
				return exp(x) / ((T)1.0 + (T)exp(x));
		});
	}

};

void resolver_default_op_Sigmoid(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Sigmoid_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Sigmoid_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Sigmoid_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
