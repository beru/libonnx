#include <onnx.h>

struct ope_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

static int ArgMin_init(onnx_node_t* n)
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

static int ArgMin_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int ArgMin_reshape(onnx_node_t* n)
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
	if (pdat->keepdims)
	{
		memcpy(&dims[0], x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}else {
		for (int i = 0, ndim = 0; i < x->ndim; i++) {
			if (i != axis)
				dims[ndim++]= x->dims[i];
		}
	}
	return y->reshape(&dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
}

static void ArgMin_int8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int8_t *p, *px = (int8_t*)x->datas;
	int8_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_int16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t *p, *px = (int16_t*)x->datas;
	int16_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_int32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int32_t *p, *px = (int32_t*)x->datas;
	int32_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_int64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t *p, *px = (int64_t*)x->datas;
	int64_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_uint8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t *p, *px = (uint8_t*)x->datas;
	uint8_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_uint16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t *p, *px = (uint16_t*)x->datas;
	uint16_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
			for (minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				if (pdat->select_last_index)
				{
					if (*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}else {
					if (*p < minv)
					{
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

static void ArgMin_uint32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint32_t *p, *px = (uint32_t*)x->datas;
	uint32_t minv;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride)
		{
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

static void ArgMin_uint64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint64_t *p, *px = (uint64_t*)x->datas;
	uint64_t minv;
	int64_t* py = (int64_t*)y->datas;
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

static void ArgMin_bfloat16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t *p, *px = (uint16_t*)x->datas;
	float minv, v;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride) {
			for (minv = bfloat16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				v = bfloat16_to_float32(*p);
				if (pdat->select_last_index) {
					if (v >= minv) {
						minv = v;
						mini = i;
					}
				}else {
					if (v > minv) {
						minv = v;
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

static void ArgMin_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t *p, *px = (uint16_t*)x->datas;
	float minv, v;
	int64_t* py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while (idx < len) {
		if (cnt < pdat->stride) {
			for (minv = float16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride) {
				v = float16_to_float32(*p);
				if (pdat->select_last_index) {
					if (v >= minv) {
						minv = v;
						mini = i;
					}
				}else {
					if (v > minv) {
						minv = v;
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

static void ArgMin_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float *p, *px = (float*)x->datas;
	float minv;
	int64_t* py = (int64_t*)y->datas;
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

static void ArgMin_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double *p, *px = (double*)x->datas;
	double minv;
	int64_t* py = (int64_t*)y->datas;
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

void resolver_default_op_ArgMin(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMin_int8,
			.int16_ = ArgMin_int16,
			.int32_ = ArgMin_int32,
			.int64_ = ArgMin_int64,
			.uint8_ = ArgMin_uint8,
			.uint16_ = ArgMin_uint16,
			.uint32_ = ArgMin_uint32,
			.uint64_ = ArgMin_uint64,
			.bfloat16_ = ArgMin_bfloat16,
			.float16_ = ArgMin_float16,
			.float32_ = ArgMin_float32,
			.float64_ = ArgMin_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMin_int8,
			.int16_ = ArgMin_int16,
			.int32_ = ArgMin_int32,
			.int64_ = ArgMin_int64,
			.uint8_ = ArgMin_uint8,
			.uint16_ = ArgMin_uint16,
			.uint32_ = ArgMin_uint32,
			.uint64_ = ArgMin_uint64,
			.float16_ = ArgMin_float16,
			.float32_ = ArgMin_float32,
			.float64_ = ArgMin_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMin_int8,
			.int16_ = ArgMin_int16,
			.int32_ = ArgMin_int32,
			.int64_ = ArgMin_int64,
			.uint8_ = ArgMin_uint8,
			.uint16_ = ArgMin_uint16,
			.uint32_ = ArgMin_uint32,
			.uint64_ = ArgMin_uint64,
			.float16_ = ArgMin_float16,
			.float32_ = ArgMin_float32,
			.float64_ = ArgMin_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.int8_ = ArgMin_int8,
			.int16_ = ArgMin_int16,
			.int32_ = ArgMin_int32,
			.int64_ = ArgMin_int64,
			.uint8_ = ArgMin_uint8,
			.uint16_ = ArgMin_uint16,
			.uint32_ = ArgMin_uint32,
			.uint64_ = ArgMin_uint64,
			.float16_ = ArgMin_float16,
			.float32_ = ArgMin_float32,
			.float64_ = ArgMin_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = ArgMin_init;
		n->exit = ArgMin_exit;
		n->reshape = ArgMin_reshape;
	}
}
