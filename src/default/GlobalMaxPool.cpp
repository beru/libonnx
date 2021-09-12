#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct GlobalMaxPool_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		for (int i = 0; i < ndim; i++) {
			if (i < 2)
				dims[i] = x->dims[i];
			else
				dims[i] = 1;
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		int N = y->dims[0];
		int C = y->dims[1];
		int m = x->strides[1];

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < C; ++j) {
				int o = i * C + j;
				py[o] = px[o * m];
				for (int k = 1; k < m; ++k)
					py[o] = max(py[o], px[o * m + k]);
			}
		}
	}

	void exec() override {
		if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_GlobalMaxPool(node_t* n)
{
	n->ope = new GlobalMaxPool_operator;
}

} // namespace onnx
