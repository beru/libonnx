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
		alpha = n->attribute("alpha", 0.0001f);
		beta = n->attribute("beta", 0.75f);
		bias = n->attribute("bias", 1.0f);
		size = n->attribute("size", 1);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
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
		if (n->opset >= 13) {
			typed_exec<LRN_operator,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<LRN_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}
};

} // namespace {

void resolver_default_op_LRN(node_t* n)
{
	n->ope = std::make_shared<LRN_operator>();
}

} // namespace onnx
