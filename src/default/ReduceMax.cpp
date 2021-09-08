#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	~operator_pdata_t() {
	}
	std::vector<int> axes;
	int naxes;
	int keepdims;
	std::vector<int> caxes;
};

bool ReduceMax_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	if (!pdat) {
		return false;
	}
	int64_t* ints;
	int nint = n->attribute("axes", &ints);
	if (nint > 0)
		pdat->naxes = nint;
	else
		pdat->naxes = n->inputs[0]->ndim;
	pdat->axes.resize(pdat->naxes);
	pdat->caxes.resize(pdat->naxes);
	if (pdat->naxes <= 0) {
		return false;
	}
	if (nint > 0) {
		for (int i = 0; i < pdat->naxes; i++)
			pdat->axes[i] = ints[i];
	}else {
		for (int i = 0; i < pdat->naxes; i++)
			pdat->axes[i] = i;
	}
	pdat->keepdims = n->attribute("keepdims", 1);
	n->priv = pdat;
	return true;
}

int ReduceMax_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
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

template <typename T>
void ReduceMax_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	T maxv, t;
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
	do {
		memset(&iter_in_axes[0], 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
		maxv = px[o];
		do {
			t = px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
			if (maxv < t)
				maxv = t;
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = maxv;
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

GEN_HOLEDR_TYPE(holder, ReduceMax_generic)

} // namespace

void resolver_default_op_ReduceMax(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int32_t, int64_t,
			uint8_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<holder,
			int8_t, int32_t, int64_t,
			uint8_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ReduceMax_init;
		n->reshape = ReduceMax_reshape;
	}
}

} // namespace onnx
