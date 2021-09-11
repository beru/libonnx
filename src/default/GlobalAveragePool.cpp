#include <onnx.h>
#include "refnd.h"
#include "util.h"

namespace onnx {

struct GlobalAveragePool_operator : public operator_t {

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
		int avgsz = x->ndata / (N * C);
		std::vector<T> buf(N * C);
		ref2d<T> sum(C, &buf[0]);
		int idx[2], cnt;
		size_t i, j, l;

		for (i = 0, l = x->ndata; i < l; i++) {
			cnt = i;
			idx[0] = cnt / x->strides[0];
			cnt %= x->strides[0];
			idx[1] = cnt / x->strides[1];
			cnt %= x->strides[1];
			sum[idx[0]][idx[1]] += px[i];
		}
		for (i = 0; i < N; i++) {
			for (j = 0; j < C; j++)
				py[i * C + j] = sum[i][j] / avgsz;
		}
	}

	void exec() override {
		if (n->opset >= 1) {
			typed_exec<GlobalAveragePool_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}

};

void resolver_default_op_GlobalAveragePool(node_t* n)
{
	n->ope = std::make_shared<GlobalAveragePool_operator>();
}

} // namespace onnx
