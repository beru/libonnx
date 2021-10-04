#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct GlobalLpPool_operator : public operator_t {
	float p;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		if (opset >= 2)
			p = attribute("p", 2);
		else
			p = attribute("p", 2.0f);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const int ndim = x->ndim;
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
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		int N = y->dims[0];
		int C = y->dims[1];
		int m = x->strides[1];

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < C; ++j) {
				int o = i * C + j;
				py[o] = 0;
				for (int k = 0; k < m; ++k)
					py[o] += pow(abs(px[o * m + k]), (T)p);
				py[o] = pow(py[o], T(1.0 / p));
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 2) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_GlobalLpPool()
{
	return new GlobalLpPool_operator;
}

} // namespace onnx
