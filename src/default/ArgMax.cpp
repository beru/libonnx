#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

namespace {

struct ope_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

int ArgMax_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->axis = n->attribute_read_int("axis", 0);
		pdat->keepdims = n->attribute_read_int("keepdims", 1);
		pdat->select_last_index = n->attribute_read_int("select_last_index", 0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

int ArgMax_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int ArgMax_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int axis = pdat->axis;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);

	if (axis < 0)
		axis += x->ndim;
	if (axis < 0 || axis >= x->ndim)
		return 0;
	pdat->dim = x->dims[axis];
	pdat->stride = x->strides[axis];
	if (pdat->keepdims)	{
		memcpy(&dims[0], x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}else {
		for (int i = 0, ndim = 0; i < x->ndim; i++)	{
			if (i != axis)
				dims[ndim++]= x->dims[i];
		}
	}
	return y->reshape(&dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
}

template <typename T>
void ArgMax_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T *p, *px = (T*)x->datas;
	T maxv;
	int64_t* py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)	{
			for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index) {
					if (*p >= maxv) {
						maxv = *p;
						maxi = i;
					}
				}else {
					if (*p > maxv) {
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}else {
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

} // namespace {

void resolver_default_op_ArgMax(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMax_generic<int8_t>,
			.int16_ = ArgMax_generic<int16_t>,
			.int32_ = ArgMax_generic<int32_t>,
			.int64_ = ArgMax_generic<int64_t>,
			.uint8_ = ArgMax_generic<uint8_t>,
			.uint16_ = ArgMax_generic<uint16_t>,
			.uint32_ = ArgMax_generic<uint32_t>,
			.uint64_ = ArgMax_generic<uint64_t>,
			.bfloat16_ = ArgMax_generic<bfloat16_t>,
			.float16_ = ArgMax_generic<float16_t>,
			.float32_ = ArgMax_generic<float>,
			.float64_ = ArgMax_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMax_generic<int8_t>,
			.int16_ = ArgMax_generic<int16_t>,
			.int32_ = ArgMax_generic<int32_t>,
			.int64_ = ArgMax_generic<int64_t>,
			.uint8_ = ArgMax_generic<uint8_t>,
			.uint16_ = ArgMax_generic<uint16_t>,
			.uint32_ = ArgMax_generic<uint32_t>,
			.uint64_ = ArgMax_generic<uint64_t>,
			.float16_ = ArgMax_generic<float16_t>,
			.float32_ = ArgMax_generic<float>,
			.float64_ = ArgMax_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMax_generic<int8_t>,
			.int16_ = ArgMax_generic<int16_t>,
			.int32_ = ArgMax_generic<int32_t>,
			.int64_ = ArgMax_generic<int64_t>,
			.uint8_ = ArgMax_generic<uint8_t>,
			.uint16_ = ArgMax_generic<uint16_t>,
			.uint32_ = ArgMax_generic<uint32_t>,
			.uint64_ = ArgMax_generic<uint64_t>,
			.float16_ = ArgMax_generic<float16_t>,
			.float32_ = ArgMax_generic<float>,
			.float64_ = ArgMax_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMax_generic<int8_t>,
			.int16_ = ArgMax_generic<int16_t>,
			.int32_ = ArgMax_generic<int32_t>,
			.int64_ = ArgMax_generic<int64_t>,
			.uint8_ = ArgMax_generic<uint8_t>,
			.uint16_ = ArgMax_generic<uint16_t>,
			.uint32_ = ArgMax_generic<uint32_t>,
			.uint64_ = ArgMax_generic<uint64_t>,
			.float16_ = ArgMax_generic<float16_t>,
			.float32_ = ArgMax_generic<float>,
			.float64_ = ArgMax_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ArgMax_init;
		n->exit = ArgMax_exit;
		n->reshape = ArgMax_reshape;
	}
}
