#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Softmax_13_operator : public operator_t {
	int axis;

	int caxis;
	int current;
	int outer;
	int inner;

	bool init() override {
		if (!is_inout_size(1, 1)) {return false;
		}
		axis = attribute("axis", -1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];

		caxis = axis;
		if (caxis < 0) {
			caxis += x->ndim;
		}
		if (caxis < 0 || caxis >= x->ndim) {
			return false;
		}
		outer = 1, inner = 1;
		for (int i = 0; i < x->ndim; ++i) {
			if (i == caxis) {
				current = x->dims[i];
			}else if (i < caxis) {
				outer *= x->dims[i];
			}else {
				inner *= x->dims[i];
			}
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (int i = 0; i < outer; ++i) {
			int oo = i * current * inner;
			for (int k = 0; k < inner; ++k) {
				int io = oo + k;
				T maxv = px[io];
				for (int j = 0; j < current; ++j) {
					int o = io + j * inner;
					if (px[o] > maxv) {
						maxv = px[o];
					}
				}
				T sum = 0;
				for (int j = 0; j < current; ++j) {
					int o = io + j * inner;
					T v = exp(px[o] - maxv);
					py[o] = v;
					sum += v;
				}
				if (sum != 0) {
					for (int j = 0; j < current; ++j) {
						io = oo + j * inner + k;
						py[io] /= sum;
					}
				}
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		typed_exec<Softmax_13_operator,
			bfloat16_t, float16_t, float, double
		>(this, type);
	}
};

struct Softmax_1_11_operator : public operator_t {
	int axis;
	int N;
	int D;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		axis = attribute("axis", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];

		if (axis < 0) {
			axis += x->ndim;
		}
		if (axis < 0 || axis >= x->ndim) {
			return false;
		}
		N = 1, D = 1;
		for (int i = 0; i < x->ndim; ++i) {
			if (i < axis) {
				N *= x->dims[i];
			}else {
				D *= x->dims[i];
			}
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (int i = 0, o = 0; i < N; i++, o += D) {
			T maxv = std::numeric_limits<T>::lowest();
			for (int j = 0; j < D; ++j) {
				if (px[o + j] > maxv) {
					maxv = px[o + j];
				}
			}
			T sum = 0;
			for (int j = 0; j < D; ++j) {
				T v = exp(px[o + j] - maxv);
				py[o + j] = v;
				sum += v;
			}
			if (sum != 0) {
				for (int j = 0; j < D; ++j) {
					py[o + j] /= sum;
				}
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		typed_exec<Softmax_1_11_operator,
			float16_t, float, double
		>(this, type);
	}

};

} // namespace {

operator_t* resolver_default_op_Softmax(int opset)
{
	if (opset >= 13) {
		return new (std::nothrow) Softmax_13_operator;
	}else {
		return new (std::nothrow) Softmax_1_11_operator;
	}
}

} // namespace onnx
