#include <onnx.h>

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
	int * shape;
	int nshape;
};

static int RandomNormal_init(onnx_node_t * n)
{
	int64_t * ints;
	int i;

	if(n->outputs.size() == 1)
	{
		operator_pdata_t * pdat = new operator_pdata_t;
		pdat->nshape = n->attribute_read_ints("shape", &ints);
		if((pdat->nshape > 0) && (pdat->shape = (int*)malloc(sizeof(int) * pdat->nshape)))
		{
			pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 1);
			pdat->mean = n->attribute_read_float("mean", 0.0);
			pdat->scale = n->attribute_read_float("scale", 1.0);
			pdat->seed = n->attribute_read_float("seed", 0.0);
			for(i = 0; i < pdat->nshape; i++)
				pdat->shape[i] = ints[i];
			n->priv = pdat;
			return 1;
		}
		else
		{
			delete pdat;
			return 0;
		}
	}
	return 0;
}

static int RandomNormal_exit(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->shape)
			free(pdat->shape);
		delete pdat;
	}
	return 1;
}

static int RandomNormal_reshape(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];

	return y->reshape(pdat->shape, pdat->nshape, pdat->dtype);
}

static void RandomNormal_operator(onnx_node_t * n)
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

void resolver_default_op_RandomNormal(onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		n->init = RandomNormal_init;
		n->exit = RandomNormal_exit;
		n->reshape = RandomNormal_reshape;
		n->ope = RandomNormal_operator;
	}
}
