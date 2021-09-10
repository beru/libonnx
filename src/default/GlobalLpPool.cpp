#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct GlobalLpPool_operator : public operator_t {
	float p;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		if (n->opset >= 2)
			p = n->attribute("p", 2);
		else
			p = n->attribute("p", 2.0f);
		return true;
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

	void exec() override {
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
				py[o] = 0;
				for (int k = 0; k < m; ++k)
					py[o] += pow(abs(px[o * m + k]), (T)p);
				py[o] = pow(py[o], T(1.0 / p));
			}
		}
	}

};

void resolver_default_op_GlobalLpPool(node_t* n)
{
	if (n->opset >= 2) {
		n->ope = ope_type_select<GlobalLpPool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<GlobalLpPool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
