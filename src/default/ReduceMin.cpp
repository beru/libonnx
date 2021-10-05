#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ReduceMin_operator : public operator_t {
	std::vector<int> axes;
	int naxes;
	int keepdims;
	std::vector<int> caxes;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		int64_t* ints;
		int nint = attribute("axes", ints);
		if (nint > 0)
			naxes = nint;
		else
			naxes = inputs[0]->ndim;
		axes.resize(naxes);
		caxes.resize(naxes);
		if (naxes <= 0) {
			return false;
		}
		if (nint > 0) {
			for (int i = 0; i < naxes; i++)
				axes[i] = ints[i];
		}else {
			for (int i = 0; i < naxes; i++)
				axes[i] = i;
		}
		keepdims = attribute("keepdims", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		for (int i = 0; i < naxes; i++) {
			int axis = axes[i];
			if (axis < 0)
				axis += x->ndim;
			if (axis < 0 || axis >= x->ndim)
				return false;
			caxes[i] = axis;
		}
		if (keepdims) {
			dims = x->dims;
			for (int i = 0; i < naxes; i++)
				dims[caxes[i]] = 1;
		}else {
			ndim = 0;
			for (int i = 0; i < x->ndim; i++) {
				bool found = false;
				for (int j = 0; j < naxes; j++) {
					if (i == caxes[j]) {
						found = true;
						break;
					}
				}
				if (!found)
					dims[ndim++]= x->dims[i];
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
		for (int i = 0; i < naxes; i++)
			mask |= (1 << caxes[i]);
		for (int i = 0, j = 0, k = 0; i < x->ndim; i++) {
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
			T minv = px[o];
			do {
				T t = px[o + dim_offset(naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
				if (minv > t)
					minv = t;
			} while (dim_next(naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
			py[i++] = minv;
		} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(type,
				int8_t, int32_t, int64_t,
				uint8_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 12) {
			TYPED_EXEC(type,
				int8_t, int32_t, int64_t,
				uint8_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 11) {
			TYPED_EXEC(type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

operator_t* resolver_default_op_ReduceMin()
{
	return new ReduceMin_operator;
}

} // namespace onnx
