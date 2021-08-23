#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	int* axes;
	int naxes;
	int keepdims;

	int* caxes;
};

bool ReduceL1_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat) {
		return false;
	}
	int64_t* ints;
	int nint = n->attribute_read_ints("axes", &ints);
	if (nint > 0)
		pdat->naxes = nint;
	else
		pdat->naxes = n->inputs[0]->ndim;
	pdat->axes = (int*)malloc(sizeof(int) * pdat->naxes);
	pdat->caxes = (int*)malloc(sizeof(int) * pdat->naxes);
	if (pdat->axes && pdat->caxes) {
		if (nint > 0) {
			for (int i = 0; i < pdat->naxes; i++)
				pdat->axes[i] = ints[i];
		}else {
			for (int i = 0; i < pdat->naxes; i++)
				pdat->axes[i] = i;
		}
		pdat->keepdims = n->attribute_read_int("keepdims", 1);
		n->priv = pdat;
		return true;
	}else {
		if (pdat->axes)
			free(pdat->axes);
		if (pdat->caxes)
			free(pdat->caxes);
		delete pdat;
		return false;
	}
}

int ReduceL1_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;

	if (pdat) {
		if (pdat->axes)
			free(pdat->axes);
		if (pdat->caxes)
			free(pdat->caxes);
		delete pdat;
	}
	return 1;
}

int ReduceL1_reshape(onnx_node_t* n)
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
		memcpy(&dims[0], x->dims, sizeof(int) * ndim);
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

template <typename T, typename SumT>
void ReduceL1_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	SumT sum;
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
			if constexpr (std::is_signed_v<T>) {
				sum += abs(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]);
			}else {
				sum += px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
			}
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = sum;
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

} // namespace

void resolver_default_op_ReduceL1(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceL1_generic<int8_t, int64_t>,
			.int32_ = ReduceL1_generic<int32_t, int64_t>,
			.int64_ = ReduceL1_generic<int64_t, int64_t>,
			.uint8_ = ReduceL1_generic<uint8_t, uint64_t>,
			.uint32_ = ReduceL1_generic<uint32_t, uint64_t>,
			.uint64_ = ReduceL1_generic<uint64_t, uint64_t>,
			.bfloat16_ = ReduceL1_generic<bfloat16_t, float>,
			.float16_ = ReduceL1_generic<float16_t, float>,
			.float32_ = ReduceL1_generic<float, float>,
			.float64_ = ReduceL1_generic<double, double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceL1_generic<int8_t, int64_t>,
			.int32_ = ReduceL1_generic<int32_t, int64_t>,
			.int64_ = ReduceL1_generic<int64_t, int64_t>,
			.uint8_ = ReduceL1_generic<uint8_t, uint64_t>,
			.uint32_ = ReduceL1_generic<uint32_t, uint64_t>,
			.uint64_ = ReduceL1_generic<uint64_t, uint64_t>,
			.float16_ = ReduceL1_generic<float16_t, float>,
			.float32_ = ReduceL1_generic<float, float>,
			.float64_ = ReduceL1_generic<double, double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceL1_generic<int8_t, int64_t>,
			.int32_ = ReduceL1_generic<int32_t, int64_t>,
			.int64_ = ReduceL1_generic<int64_t, int64_t>,
			.uint8_ = ReduceL1_generic<uint8_t, uint64_t>,
			.uint32_ = ReduceL1_generic<uint32_t, uint64_t>,
			.uint64_ = ReduceL1_generic<uint64_t, uint64_t>,
			.float16_ = ReduceL1_generic<float16_t, float>,
			.float32_ = ReduceL1_generic<float, float>,
			.float64_ = ReduceL1_generic<double, double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ReduceL1_init;
		n->exit = ReduceL1_exit;
		n->reshape = ReduceL1_reshape;
	}
}
