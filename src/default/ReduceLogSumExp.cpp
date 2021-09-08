#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	~operator_pdata_t() {
	}
	std::vector<int> axes;
	int naxes;
	int keepdims;
	std::vector<int> caxes;
};

bool ReduceLogSumExp_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat) {
		return false;
	}
	int64_t* ints;
	int nint = n->read_attribute("axes", &ints);
	if (nint > 0)
		pdat->naxes = nint;
	else
		pdat->naxes = n->inputs[0]->ndim;
	pdat->axes.resize(pdat->naxes);
	pdat->caxes.resize(pdat->naxes);
	if (pdat->naxes > 0) {
		if (nint > 0) {
			for (int i = 0; i < pdat->naxes; i++)
				pdat->axes[i] = ints[i];
		}else {
			for (int i = 0; i < pdat->naxes; i++)
				pdat->axes[i] = i;
		}
		pdat->keepdims = n->read_attribute("keepdims", 1);
		n->priv = pdat;
		return true;
	}else {
		delete pdat;
		return false;
	}
}

int ReduceLogSumExp_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int axis, found;
	int i, j;

	for (i = 0; i < pdat->naxes; i++) {
		axis = pdat->axes[i];
		if (axis < 0)
			axis += x->ndim;
		if (axis < 0 || axis >= x->ndim)
			return 0;
		pdat->caxes[i] = axis;
	}
	if (pdat->keepdims) {
		dims = x->dims;
		for (i = 0; i < pdat->naxes; i++)
			dims[pdat->caxes[i]] = 1;
	}else {
		for (i = 0, ndim = 0; i < x->ndim; i++) {
			for (j = 0, found = 0; j < pdat->naxes; j++) {
				if (i == pdat->caxes[j]) {
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

template <typename T> struct SumType {};
#define X(t0, t1) template <> struct SumType<t0> { using type = t1; };
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
void ReduceLogSumExp_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	typename SumType<T>::type sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	std::vector<int> iter_not_in_axes_max(not_in_axes_num);
	std::vector<int> iter_not_in_axes(not_in_axes_num);
	std::vector<int> not_in_axes_axis_dis(x->ndim);
	std::vector<int> iter_in_axes_max(pdat->naxes);
	std::vector<int> in_axes_axis_dis(pdat->naxes);
	std::vector<int> iter_in_axes(pdat->naxes);
	uint32_t mask;
	int i, j, k, o;

	for (i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
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
	memset(&iter_not_in_axes[0], 0, sizeof(int) * not_in_axes_num);
	do {
		memset(&iter_in_axes[0], 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
		sum = 0;
		do {
			sum += expf(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]);
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = logf(sum);
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

GEN_HOLEDR_TYPE(holder, ReduceLogSumExp_generic)

} // namespace

void resolver_default_op_ReduceLogSumExp(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint32_t, uint64_t,
			int8_t, int32_t, int64_t,
			float16_t, float, double, bfloat16_t
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint32_t, uint64_t,
			int8_t, int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint32_t, uint64_t,
			int8_t, int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ReduceLogSumExp_init;
		n->reshape = ReduceLogSumExp_reshape;
	}
}
