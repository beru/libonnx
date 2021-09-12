#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ReduceProd_operator : public operator_t {
	std::vector<int> axes;
	int naxes;
	int keepdims;
	std::vector<int> caxes;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		int64_t* ints;
		int nint = n->attribute("axes", &ints);
		if (nint > 0)
			naxes = nint;
		else
			naxes = n->inputs[0]->ndim;
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
		keepdims = n->attribute("keepdims", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);
		int axis, found;
		int i, j;

		for (i = 0; i < naxes; i++) {
			axis = axes[i];
			if (axis < 0)
				axis += x->ndim;
			if (axis < 0 || axis >= x->ndim)
				return false;
			caxes[i] = axis;
		}
		if (keepdims) {
			dims = x->dims;
			for (i = 0; i < naxes; i++)
				dims[caxes[i]] = 1;
		}else {
			for (i = 0, ndim = 0; i < x->ndim; i++) {
				for (j = 0, found = 0; j < naxes; j++) {
					if (i == caxes[j]) {
						found = 1;
						break;
					}
				}
				if (!found)
					dims[ndim++]= x->dims[i];
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T> struct ProdType {};
	#define X(t0, t1) template <> struct ProdType<t0> { using type = t1; };
	X(int8_t, float)
	X(int32_t, float)
	X(int64_t, float)
	X(uint8_t, float)
	X(uint32_t, float)
	X(uint64_t, float)
	X(bfloat16_t, float)
	X(float16_t, float)
	X(float, float)
	X(double, double)
	#undef X

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		typename ProdType<T>::type prod;
		int not_in_axes_num = x->ndim - naxes;
		std::vector<int> iter_not_in_axes_max(not_in_axes_num);
		std::vector<int> iter_not_in_axes(not_in_axes_num);
		std::vector<int> not_in_axes_axis_dis(x->ndim);
		std::vector<int> iter_in_axes_max(naxes);
		std::vector<int> in_axes_axis_dis(naxes);
		std::vector<int> iter_in_axes(naxes);
		uint32_t mask;
		int i, j, k, o;

		for (i = 0, mask = 0; i < naxes; i++)
			mask |= (1 << caxes[i]);
		for (i = 0, j = 0, k = 0; i < x->ndim; i++) {
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
		i = 0;
		do {
			std::fill(iter_in_axes.begin(), iter_in_axes.end(), 0);
			o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
			prod = 1;
			do {
				prod *= px[o + dim_offset(naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
			} while (dim_next(naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
			py[i++] = prod;
		} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				uint8_t, uint32_t, uint64_t,
				int8_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t
			)
		}else if (n->opset >= 11) {
			TYPED_EXEC(n->inputs[0]->type,
				uint8_t, uint32_t, uint64_t,
				int8_t, int32_t, int64_t,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				uint8_t, uint32_t, uint64_t,
				int8_t, int32_t, int64_t,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_ReduceProd(node_t* n)
{
	n->ope = std::make_shared<ReduceProd_operator>();
}

} // namespace onnx
