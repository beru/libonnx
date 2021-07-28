#include <onnx.h>

struct ope_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

static int ArgMax_init(onnx_node_t * n)
{
	ope_pdata_t * pdat;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = (ope_pdata_t *)malloc(sizeof(ope_pdata_t));
		if(pdat)
		{
			pdat->axis = onnx_attribute_read_int(n, "axis", 0);
			pdat->keepdims = onnx_attribute_read_int(n, "keepdims", 1);
			pdat->select_last_index = onnx_attribute_read_int(n, "select_last_index", 0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int ArgMax_exit(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int ArgMax_reshape(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int axis = pdat->axis;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	if(axis < 0)
		axis += x->ndim;
	if(axis < 0 || axis >= x->ndim)
		return 0;
	pdat->dim = x->dims[axis];
	pdat->stride = x->strides[axis];
	if(pdat->keepdims)
	{
		memcpy(&dims[0], x->dims, sizeof(int) * ndim);
		dims[axis] = 1;
	}
	else
	{
		for(i = 0, ndim = 0; i < x->ndim; i++)
		{
			if(i != axis)
				dims[ndim++]= x->dims[i];
		}
	}
	return onnx_tensor_reshape(y, &dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
}

static void ArgMax_int8(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int8_t * p, * px = (int8_t*)x->datas;
	int8_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int16_t * p, * px = (int16_t*)x->datas;
	int16_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int32_t * p, * px = (int32_t*)x->datas;
	int32_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_int64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int64_t * p, * px = (int64_t*)x->datas;
	int64_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint8(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint8_t * p, * px = (uint8_t*)x->datas;
	uint8_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	uint16_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint32_t * p, * px = (uint32_t*)x->datas;
	uint32_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_uint64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint64_t * p, * px = (uint64_t*)x->datas;
	uint64_t maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_bfloat16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	float maxv, v;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = bfloat16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = bfloat16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
				else
				{
					if(v > maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	float maxv, v;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = float16_to_float32(px[idx]), maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = float16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
				else
				{
					if(v > maxv)
					{
						maxv = v;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	float * p, * px = (float*)x->datas;
	float maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

static void ArgMax_float64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	double * p, * px = (double*)x->datas;
	double maxv;
	int64_t * py = (int64_t*)y->datas;
	int64_t maxi;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(maxv = px[idx], maxi = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p >= maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
				else
				{
					if(*p > maxv)
					{
						maxv = *p;
						maxi = i;
					}
				}
			}
			*py++ = maxi;
			idx++;
			cnt++;
		}
		else
		{
			idx += (pdat->dim - 1) * pdat->stride;
			cnt = 0;
		}
	}
}

void resolver_default_op_ArgMax(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMax_init;
			n->exit = ArgMax_exit;
			n->reshape = ArgMax_reshape;
			n->ope = ArgMax_float64;
			break;
		default:
			break;
		}
	}
}
