#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
};

bool RandomNormalLike_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->dtype = (tensor_type_t)n->read_attribute("dtype", ONNX_TENSOR_TYPE_UNDEFINED);
	pdat->mean = n->read_attribute("mean", 0.0f);
	pdat->scale = n->read_attribute("scale", 1.0f);
	pdat->seed = n->read_attribute("seed", 0.0f);
	n->priv = pdat;
	return true;
}

int RandomNormalLike_reshape(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	tensor_type_t type;

	if (pdat->dtype != ONNX_TENSOR_TYPE_UNDEFINED)
		type = pdat->dtype;
	else
		type = x->type;
	switch (type) {
	case ONNX_TENSOR_TYPE_FLOAT16:
	case ONNX_TENSOR_TYPE_FLOAT32:
	case ONNX_TENSOR_TYPE_FLOAT64:
		return y->reshape(&x->dims[0], x->ndim, type);
	default:
		break;
	}
	return 0;
}

void RandomNormalLike_operator(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* y = n->outputs[0];

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

void resolver_default_op_RandomNormalLike(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomNormalLike_operator;
	}
	if (n->ope) {
		n->init = RandomNormalLike_init;
		n->reshape = RandomNormalLike_reshape;
	}
}

} // namespace onnx
