#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Abs_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}
	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if constexpr (std::is_signed_v<T>) {
				py[i] = abs(px[i]);
			}else {
				py[i] = px[i];
			}
		}
	}
};
	
void resolver_default_op_Abs(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Abs_operator,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double, bfloat16_t
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Abs_operator,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Abs_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
