#include <onnx.h>

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
};

static int RandomNormalLike_init(onnx_node_t * n)
{
	operator_pdata_t * pdat;

	if((n->inputs.size() == 1) && (n->outputs.size() == 1))
	{
		pdat = (operator_pdata_t *)malloc(sizeof(operator_pdata_t));
		if(pdat)
		{
			pdat->dtype = (onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 0);
			pdat->mean = onnx_attribute_read_float(n, "mean", 0.0);
			pdat->scale = onnx_attribute_read_float(n, "scale", 1.0);
			pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int RandomNormalLike_exit(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int RandomNormalLike_reshape(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_type_t type;

	if(pdat->dtype != ONNX_TENSOR_TYPE_UNDEFINED)
		type = pdat->dtype;
	else
		type = x->type;
	switch(type)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT32:
	case ONNX_TENSOR_TYPE_FLOAT64:
		return onnx_tensor_reshape(y, x->dims, x->ndim, type);
	default:
		break;
	}
	return 0;
}

static void RandomNormalLike_operator(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];

	if(pdat->seed != 0.0)
		srand(pdat->seed);
	switch(pdat->dtype)
	{
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * py = (uint16_t *)y->datas;
			float ty, tx;
			for(size_t i = 0, l = y->ndata; i < l; i++)
			{
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = float16_to_float32(pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty));
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * py = (float *)y->datas;
			float ty, tx;
			for(size_t i = 0, l = y->ndata; i < l; i++)
			{
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty);
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * py = (double *)y->datas;
			double ty, tx;
			for(size_t i = 0, l = y->ndata; i < l; i++)
			{
				ty = (double)rand() / (RAND_MAX + 1.0f);
				tx = (double)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrt(-2.0f * log(tx)) * cos(2.0f * acos(-1.0f) * ty);
			}
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_RandomNormalLike(onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		n->init = RandomNormalLike_init;
		n->exit = RandomNormalLike_exit;
		n->reshape = RandomNormalLike_reshape;
		n->ope = RandomNormalLike_operator;
	}
}
