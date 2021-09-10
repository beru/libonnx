#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Erf_operator : public operator_t {
	bool init() override {
		return is_inout_size(n, 1, 1);
	};
	void exec() override {
		foreach_tensor<T>(n, [](auto x){return erf(x);});
	}
};

void resolver_default_op_Erf(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Erf_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<Erf_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
