#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Abs_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
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

	void exec() override {
		tensor_type_t type = n->inputs[0]->type;
		if (n->opset >= 13) {
			typed_exec<Abs_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t
			>(type);
		}else if (n->opset >= 6) {
			typed_exec<Abs_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double
			>(type);
		}else if (n->opset >= 1) {
			typed_exec<Abs_operator,
				float16_t, float, double
			>(type);
		}
	}
};

} // namespace {

void resolver_default_op_Abs(node_t* n)
{
	n->ope = std::make_shared<Abs_operator>();
}

} // namespace onnx
