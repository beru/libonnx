#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
};

bool RandomNormalLike_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 0);
	pdat->mean = n->attribute_read_float("mean", 0.0);
	pdat->scale = n->attribute_read_float("scale", 1.0);
	pdat->seed = n->attribute_read_float("seed", 0.0);
	n->priv = pdat;
	return true;
}

int RandomNormalLike_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int RandomNormalLike_reshape(onnx_node_t* n)
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

void RandomNormalLike_operator(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	if (pdat->seed != 0.0)
		srand(pdat->seed);
	switch (pdat->dtype) {
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			float ty, tx;
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty);
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			float ty, tx;
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				ty = (float)rand() / (RAND_MAX + 1.0f);
				tx = (float)rand() / (RAND_MAX + 1.0f);
				py[i] = pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty);
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			double ty, tx;
			for (size_t i = 0, l = y->ndata; i < l; i++) {
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

} // namespace

void resolver_default_op_RandomNormalLike(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomNormalLike_operator;
	}
	if (n->ope) {
		n->init = RandomNormalLike_init;
		n->exit = RandomNormalLike_exit;
		n->reshape = RandomNormalLike_reshape;
	}
}
