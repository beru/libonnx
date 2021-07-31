#include <onnx.h>

struct ope_pdata_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;
};

static int ArgMin_init(onnx_node_t * n)
{

	if((n->inputs.size() == 1) && (n->outputs.size() == 1))
	{
		ope_pdata_t * pdat = new ope_pdata_t;
		pdat->axis = onnx_attribute_read_int(n, "axis", 0);
		pdat->keepdims = onnx_attribute_read_int(n, "keepdims", 1);
		pdat->select_last_index = onnx_attribute_read_int(n, "select_last_index", 0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int ArgMin_exit(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	delete pdat;
	return 1;
}

static int ArgMin_reshape(onnx_node_t * n)
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

static void ArgMin_int8(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int8_t * p, * px = (int8_t*)x->datas;
	int8_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int16_t * p, * px = (int16_t*)x->datas;
	int16_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int32_t * p, * px = (int32_t*)x->datas;
	int32_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_int64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int64_t * p, * px = (int64_t*)x->datas;
	int64_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint8(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint8_t * p, * px = (uint8_t*)x->datas;
	uint8_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	uint16_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint32_t * p, * px = (uint32_t*)x->datas;
	uint32_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_uint64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint64_t * p, * px = (uint64_t*)x->datas;
	uint64_t minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_bfloat16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	float minv, v;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = bfloat16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = bfloat16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= minv)
					{
						minv = v;
						mini = i;
					}
				}
				else
				{
					if(v > minv)
					{
						minv = v;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float16(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * p, * px = (uint16_t*)x->datas;
	float minv, v;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = float16_to_float32(px[idx]), mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				v = float16_to_float32(*p);
				if(pdat->select_last_index)
				{
					if(v >= minv)
					{
						minv = v;
						mini = i;
					}
				}
				else
				{
					if(v > minv)
					{
						minv = v;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float32(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	float * p, * px = (float*)x->datas;
	float minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

static void ArgMin_float64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	double * p, * px = (double*)x->datas;
	double minv;
	int64_t * py = (int64_t*)y->datas;
	int64_t mini;
	size_t len = x->ndata;
	size_t idx = 0;
	int cnt = 0;
	int i;

	while(idx < len)
	{
		if(cnt < pdat->stride)
		{
			for(minv = px[idx], mini = 0, i = 1, p = px + idx + pdat->stride; i < pdat->dim; i++, p += pdat->stride)
			{
				if(pdat->select_last_index)
				{
					if(*p <= minv)
					{
						minv = *p;
						mini = i;
					}
				}
				else
				{
					if(*p < minv)
					{
						minv = *p;
						mini = i;
					}
				}
			}
			*py++ = mini;
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

void resolver_default_op_ArgMin(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float64;
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
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float64;
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
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float64;
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
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ArgMin_init;
			n->exit = ArgMin_exit;
			n->reshape = ArgMin_reshape;
			n->ope = ArgMin_float64;
			break;
		default:
			break;
		}
	}
}
