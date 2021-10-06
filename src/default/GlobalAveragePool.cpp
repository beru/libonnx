#include "onnx.h"
#include "refnd.h"
#include "util.h"

namespace onnx {

namespace {

struct GlobalAveragePool_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const int ndim = x->ndim;
		std::vector<int> dims(ndim);

		for (int i = 0; i < ndim; i++) {
			if (i < 2) {
				dims[i] = x->dims[i];
			}else {
				dims[i] = 1;
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		int N = y->dims[0];
		int C = y->dims[1];
		const int avgsz = x->ndata / (N * C);
		std::vector<T> buf(N * C);
		ref2d<T> sum(C, &buf[0]);
		size_t l = x->ndata;
		for (size_t i = 0; i < l; i++) {
			int cnt = i;
			int idx0 = cnt / x->strides[0];
			cnt %= x->strides[0];
			int idx1 = cnt / x->strides[1];
			cnt %= x->strides[1];
			sum[idx0][idx1] += px[i];
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < C; j++) {
				py[i * C + j] = sum[i][j] / avgsz;
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 1) {
			typed_exec<GlobalAveragePool_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_GlobalAveragePool(int opset)
{
	return new GlobalAveragePool_operator;
}

} // namespace onnx
