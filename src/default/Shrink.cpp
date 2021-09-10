#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Shrink_operator : public operator_t {
	float bias;
	float lambd;

	bool init() override {
		if (!is_inout_size(n, 1, 1)) {
			return false;
		}
		bias = n->attribute("bias", 0.0f);
		lambd = n->attribute("lambd", 0.5f);
		return true;
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] < -lambd)
				py[i] = px[i] + (T)bias;
			else if (px[i] > lambd)
				py[i] = px[i] - (T)bias;
			else
				py[i] = 0;
		}
	}

};

void resolver_default_op_Shrink(node_t* n)
{
	if (n->opset >= 9) {
		n->ope = ope_type_select<Shrink_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
