#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	tensor_type_t dtype;
	float high;
	float low;
	float seed;
};

bool RandomUniformLike_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->dtype = (tensor_type_t)n->read_attribute("dtype", ONNX_TENSOR_TYPE_UNDEFINED);
	pdat->high = n->read_attribute("high", 1.0f);
	pdat->low = n->read_attribute("low", 0.0f);
	pdat->seed = n->read_attribute("seed", 0.0f);
	n->priv = pdat;
	return true;
}

int RandomUniformLike_reshape(node_t* n)
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

template <typename T>
void RandomUniformLike(T* py, size_t ndata, float high, float low)
{
	for (size_t i = 0; i < ndata; i++)
		py[i] = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
}

void RandomUniformLike_operator(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* y = n->outputs[0];

	if (pdat->seed != 0.0)
		srand(pdat->seed);
	switch (pdat->dtype) {
	case ONNX_TENSOR_TYPE_FLOAT16:
		RandomUniformLike((float16_t*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		RandomUniformLike((float*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		RandomUniformLike((double*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	default:
		break;
	}
}

} // namespace

void resolver_default_op_RandomUniformLike(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomUniformLike_operator;
	}
	if (n->ope) {
		n->init = RandomUniformLike_init;
		n->reshape = RandomUniformLike_reshape;
	}
}

} // namespace onnx
