#include "onnx.h"
#include "util.h"

namespace onnx {

template <typename T> struct SumType {};
#define X(t0, t1) template <> struct SumType<t0> { using type = t1; };
X(int8_t, int64_t)
X(int32_t, int64_t)
X(int64_t, int64_t)
X(uint8_t, uint64_t)
X(uint32_t, uint64_t)
X(uint64_t, uint64_t)
X(bfloat16_t, float)
X(float16_t, float)
X(float, float)
X(double, double)
#undef X

namespace {

struct ReduceSum_operator : public operator_t {
	int keepdims;
	int noop_with_empty_axes;

	int caxes[32];
	int naxes;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		keepdims = attribute("keepdims", 1);
		noop_with_empty_axes = attribute("noop_with_empty_axes", 0);
		return true;
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		if ((inputs.size() > 1) && (inputs[1]->ndata > 0)) {
			const tensor_t* a = inputs[1];
			const int64_t* pa = (const int64_t*)a->data;
			naxes = min(min(x->ndim, 32), (int)a->ndata);
			for (int i = 0; i < naxes; ++i) {
				int axis = pa[i];
				if (axis < 0) {
					axis += x->ndim;
				}
				if (axis < 0 || axis >= x->ndim) {
					return false;
				}
				caxes[i] = axis;
			}
		}else if (noop_with_empty_axes == 0) {
			naxes = min(x->ndim, 32);
			for (int i = 0; i < naxes; ++i) {
				caxes[i] = i;
			}
		}else {
			naxes = 0;
		}
		if (keepdims) {
			dims = x->dims;
			for (int i = 0; i < naxes; ++i) {
				dims[caxes[i]] = 1;
			}
		}else {
			ndim = 0;
			for (int i = 0; i < x->ndim; ++i) {
				bool found = false;
				for (int j = 0; j < naxes; ++j) {
					if (i == caxes[j]) {
						found = true;
						break;
					}
				}
				if (!found) {
					dims[ndim++]= x->dims[i];
				}
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
		int not_in_axes_num = x->ndim - naxes;
		std::vector<int> iter_not_in_axes_max(not_in_axes_num);
		std::vector<int> iter_not_in_axes(not_in_axes_num);
		std::vector<int> not_in_axes_axis_dis(x->ndim);
		std::vector<int> iter_in_axes_max(naxes);
		std::vector<int> in_axes_axis_dis(naxes);
		std::vector<int> iter_in_axes(naxes);
		uint32_t mask = 0;
		for (int i = 0; i < naxes; ++i) {
			mask |= (1 << caxes[i]);
		}
		for (int i = 0, j = 0, k = 0; i < x->ndim; ++i) {
			if (mask & (1 << i)) {
				in_axes_axis_dis[j] = x->strides[i];
				iter_in_axes_max[j] = x->dims[i];
				j += 1;
				continue;
			}
			not_in_axes_axis_dis[k] = x->strides[i];
			iter_not_in_axes_max[k] = x->dims[i];
			k += 1;
		}
		int i = 0;
		do {
			std::fill(iter_in_axes.begin(), iter_in_axes.end(), 0);
			int o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
			typename SumType<T>::type sum = 0;
			do {
				sum += px[o + dim_offset(naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
			} while (dim_next(naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
			py[i++] = sum;
		} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<ReduceSum_operator,
				int8_t, int32_t, int64_t,
				uint8_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 11) {
			typed_exec<ReduceSum_operator,
				int8_t, int32_t, int64_t,
				uint8_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<ReduceSum_operator,
				int8_t, int32_t, int64_t,
				uint8_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_ReduceSum(int opset)
{
	return new ReduceSum_operator;
}

} // namespace onnx
