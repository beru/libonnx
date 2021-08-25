#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	float high;
	float low;
	float seed;
};

bool RandomUniformLike_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 0);
	pdat->high = n->attribute_read_float("high", 1.0);
	pdat->low = n->attribute_read_float("low", 0.0);
	pdat->seed = n->attribute_read_float("seed", 0.0);
	n->priv = pdat;
	return true;
}

int RandomUniformLike_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int RandomUniformLike_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_type_t type;

	if (pdat->dtype != ONNX_TENSOR_TYPE_UNDEFINED)
		type = pdat->dtype;
	else
		type = x->type;
	switch (type) {
	case ONNX_TENSOR_TYPE_FLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT32:
	case ONNX_TENSOR_TYPE_FLOAT64:
		return y->reshape(x->dims, x->ndim, type);
	default:
		break;
	}
	return 0;
}

void RandomUniformLike_operator(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	if (pdat->seed != 0.0)
		srand(pdat->seed);
	switch (pdat->dtype) {
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (size_t i = 0, l = y->ndata; i < l; i++)
				py[i] = ((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (size_t i = 0, l = y->ndata; i < l; i++)
				py[i] = ((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (size_t i = 0, l = y->ndata; i < l; i++)
				py[i] = ((double)rand() / (double)RAND_MAX) * (pdat->high - pdat->low) + pdat->low;
		}
		break;
	default:
		break;
	}
}

} // namespace

void resolver_default_op_RandomUniformLike(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomUniformLike_operator;
	}
	if (n->ope) {
		n->init = RandomUniformLike_init;
		n->exit = RandomUniformLike_exit;
		n->reshape = RandomUniformLike_reshape;
	}
}
