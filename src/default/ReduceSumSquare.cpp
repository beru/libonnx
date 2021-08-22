#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

struct ope_pdata_t {
	int* axes;
	int naxes;
	int keepdims;

	int* caxes;
};

static int ReduceSumSquare_init(onnx_node_t* n)
{
	int64_t* ints;
	int nint;
	int i;

	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		nint = n->attribute_read_ints("axes", &ints);
		if (nint > 0)
			pdat->naxes = nint;
		else
			pdat->naxes = n->inputs[0]->ndim;
		pdat->axes = (int*)malloc(sizeof(int) * pdat->naxes);
		pdat->caxes = (int*)malloc(sizeof(int) * pdat->naxes);
		if (pdat->axes && pdat->caxes) {
			if (nint > 0) {
				for (i = 0; i < pdat->naxes; i++)
					pdat->axes[i] = ints[i];
			}else {
				for (i = 0; i < pdat->naxes; i++)
					pdat->axes[i] = i;
			}
			pdat->keepdims = n->attribute_read_int("keepdims", 1);
			n->priv = pdat;
			return 1;
		}else {
			if (pdat->axes)
				free(pdat->axes);
			if (pdat->caxes)
				free(pdat->caxes);
			delete pdat;
		}
	}
	return 0;
}

static int ReduceSumSquare_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;

	if (pdat) {
		if (pdat->axes)
			free(pdat->axes);
		if (pdat->caxes)
			free(pdat->caxes);
		delete pdat;
	}
	return 1;
}

static int ReduceSumSquare_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
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

template <typename T> struct SumType {};
template <> struct SumType<int8_t> { using type = int64_t; };
template <> struct SumType<int16_t> { using type = int64_t; };
template <> struct SumType<int32_t> { using type = int64_t; };
template <> struct SumType<int64_t> { using type = int64_t; };
template <> struct SumType<uint8_t> { using type = uint64_t; };
template <> struct SumType<uint16_t> { using type = uint64_t; };
template <> struct SumType<uint32_t> { using type = uint64_t; };
template <> struct SumType<uint64_t> { using type = uint64_t; };
template <> struct SumType<bfloat16_t> { using type = float; };
template <> struct SumType<float16_t> { using type = float; };
template <> struct SumType<float> { using type = float; };
template <> struct SumType<double> { using type = double; };

template <typename T>
static void ReduceSumSquare_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	T v;
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
			v = px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
			sum += v * v;
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = sum;
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

void resolver_default_op_ReduceSumSquare(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceSumSquare_generic<int8_t>,
			.int32_ = ReduceSumSquare_generic<int32_t>,
			.int64_ = ReduceSumSquare_generic<int64_t>,
			.uint8_ = ReduceSumSquare_generic<uint8_t>,
			.uint32_ = ReduceSumSquare_generic<uint32_t>,
			.uint64_ = ReduceSumSquare_generic<uint64_t>,
			.bfloat16_ = ReduceSumSquare_generic<bfloat16_t>,
			.float16_ = ReduceSumSquare_generic<float16_t>,
			.float32_ = ReduceSumSquare_generic<float>,
			.float64_ = ReduceSumSquare_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceSumSquare_generic<int8_t>,
			.int32_ = ReduceSumSquare_generic<int32_t>,
			.int64_ = ReduceSumSquare_generic<int64_t>,
			.uint8_ = ReduceSumSquare_generic<uint8_t>,
			.uint32_ = ReduceSumSquare_generic<uint32_t>,
			.uint64_ = ReduceSumSquare_generic<uint64_t>,
			.float16_ = ReduceSumSquare_generic<float16_t>,
			.float32_ = ReduceSumSquare_generic<float>,
			.float64_ = ReduceSumSquare_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ReduceSumSquare_generic<int8_t>,
			.int32_ = ReduceSumSquare_generic<int32_t>,
			.int64_ = ReduceSumSquare_generic<int64_t>,
			.uint8_ = ReduceSumSquare_generic<uint8_t>,
			.uint32_ = ReduceSumSquare_generic<uint32_t>,
			.uint64_ = ReduceSumSquare_generic<uint64_t>,
			.float16_ = ReduceSumSquare_generic<float16_t>,
			.float32_ = ReduceSumSquare_generic<float>,
			.float64_ = ReduceSumSquare_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ReduceSumSquare_init;
		n->exit = ReduceSumSquare_exit;
		n->reshape = ReduceSumSquare_reshape;
	}
}
