#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct LogSoftmax_13_operator : public operator_t {
	int axis;
	int caxis;
	int current;
	int outter;
	int inner;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		axis = n->attribute("axis", -1);
		return true;
	}

	bool reshape() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int i;

		caxis = axis;
		if (caxis < 0)
			caxis += x->ndim;
		if (caxis < 0 || caxis >= x->ndim)
			return false;
		for (i = 0, outter = 1, inner = 1; i < x->ndim; i++) {
			if (i == caxis)
				current = x->dims[i];
			else if (i < caxis)
				outter *= x->dims[i];
			else
				inner *= x->dims[i];
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T maxv, sum;
		int i, j, k, o, oo, io;

		for (i = 0; i < outter; i++) {
			oo = i * current * inner;
			for (k = 0; k < inner; k++) {
				io = oo + k;
				for (j = 0, maxv = px[io]; j < current; j++) {
					o = io + j * inner;
					if (px[o] > maxv)
						maxv = px[o];
				}
				for (j = 0, sum = 0; j < current; j++) {
					o = io + j * inner;
					py[o] = exp(px[o] - maxv);
					sum += py[o];
				}
				if (sum != 0) {
					for (j = 0; j < current; j++) {
						io = oo + j * inner + k;
						py[io] = log(py[io] / sum);
					}
				}
			}
		}
	}

	void exec() override {
		typed_exec<LogSoftmax_13_operator,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}
};

struct LogSoftmax_1_11_operator : public operator_t {
	int axis;
	int N;
	int D;

	bool init() override {
		if (!(n->inputs.size() == 1 && n->outputs.size() == 1)) {
			return false;
		}
		axis = n->attribute("axis", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int i;

		if (axis < 0)
			axis += x->ndim;
		if (axis < 0 || axis >= x->ndim)
			return false;
		for (i = 0, N = 1, D = 1; i < x->ndim; i++) {
			if (i < axis)
				N *= x->dims[i];
			else
				D *= x->dims[i];
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T maxv, sum;
		int i, j, o;

		for (i = 0, o = 0; i < N; i++, o += D) {
			for (j = 0, maxv = std::numeric_limits<T>::min(); j < D; j++) {
				if (px[o + j] > maxv)
					maxv = px[o + j];
			}
			for (j = 0, sum = 0; j < D; j++) {
				py[o + j] = exp(px[o + j] - maxv);
				sum += py[o + j];
			}
			if (sum != 0) {
				for (j = 0; j < D; j++)
					py[o + j] = log(py[o + j] / sum);
			}
		}
	}

	void exec() override {
		TYPED_EXEC(n->inputs[0]->type,
			float16_t, float, double
		)
	}
};

} // namespace {

void resolver_default_op_LogSoftmax(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = new LogSoftmax_13_operator;
	}else {
		n->ope = new LogSoftmax_1_11_operator;
	}
}

} // namespace onnx
