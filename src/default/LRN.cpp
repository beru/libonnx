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
		T sum, t;
		T over = alpha / size;
		int N = x->dims[0];
		int C = x->dims[1];
		int L = x->strides[1];
		int start, end;
		int i, j, u, v, o;

		for (u = 0; u < N; u++) {
			for (v = 0; v < C; v++) {
				for (i = 0; i < L; i++) {
					start = v - (size / 2);
					if (start < 0)
						start = 0;
					end = v + (size / 2);
					if (end >= C)
						end = C - 1;
					for (j = start, sum = 0; j <= end; ++j) {
						t = px[(u * C + j) * L + i];
						sum += t * t;
					}
					o = (u * C + v) * L + i;
					py[o] = px[o] * pow(bias + over * sum, -beta);
				}
			}
		}
	}

	void exec() override {
		if (opset >= 13) {
			TYPED_EXEC(inputs[0]->type,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(inputs[0]->type,
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
