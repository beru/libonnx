#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct LRN_operator : public operator_t {
	float alpha;
	float beta;
	float bias;
	int size;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 0.0001f);
		beta = attribute("beta", 0.75f);
		bias = attribute("bias", 1.0f);
		size = attribute("size", 1);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		const T over = alpha / size;
		const int N = x->dims[0];
		const int C = x->dims[1];
		const int L = x->strides[1];
		for (int u = 0; u < N; u++) {
			for (int v = 0; v < C; v++) {
				for (int i = 0; i < L; i++) {
					int start = v - (size / 2);
					if (start < 0)
						start = 0;
					int end = v + (size / 2);
					if (end >= C)
						end = C - 1;
					T sum = 0;
					for (int j = start; j <= end; ++j) {
						T t = px[(u * C + j) * L + i];
						sum += t * t;
					}
					int o = (u * C + v) * L + i;
					py[o] = px[o] * pow(bias + over * sum, -beta);
				}
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(type,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

operator_t* resolver_default_op_LRN()
{
	return new LRN_operator;
}

} // namespace onnx
