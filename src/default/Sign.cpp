#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Sign_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		foreach_tensor<T>(n, [](auto x){
			if (x > 0)
				return 1;
			else if (x < 0)
				return -1;
			else
				return 0;
		});
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 9) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_Sign(node_t* n)
{
	n->ope = std::make_shared<Sign_operator>();
}

} // namespace onnx
