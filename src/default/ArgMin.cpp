#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

bool ArgMin_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	if (!pdat)
		return false;
	pdat->axis = n->attribute("axis", 0);
	pdat->keepdims = n->attribute("keepdims", 1);
	pdat->select_last_index = n->attribute("select_last_index", 0);
	n->priv = pdat;
	return true;
}

int ArgMin_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int axis = pdat->axis;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);

	if (axis < 0)
		axis += x->ndim;
	if (axis < 0 || axis >= x->ndim)
		return 0;
	pdat->dim = x->dims[axis];
	pdat->stride = x->strides[axis];
	if (pdat->keepdims) {
		dims = x->dims;
		dims[axis] = 1;
	}else {
		for (int i = 0, ndim = 0; i < x->ndim; i++) {
			if (i != axis)
				dims[ndim++]= x->dims[i];
		}
	}
	return y->reshape(&dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
}

template <typename T>
void ArgMin_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T *p, *px = (T*)x->data;
	T minv;
	int64_t* py = (int64_t*)y->data;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride) {
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index) {
					if (*p <= minv) {
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv) {
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
			idx++;
			cnt++;
		}else {
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

GEN_HOLEDR_TYPE(holder, ArgMin_generic)

} // namespace

void resolver_default_op_ArgMin(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ArgMin_init;
		n->reshape = ArgMin_reshape;
	}
}

} // namespace onnx
