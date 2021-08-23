#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	float high;
	float low;
	float seed;
	int* shape;
	int nshape;
};

bool RandomUniform_init(onnx_node_t* n)
{
	if (n->outputs.size() != 1) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	int64_t* ints;
	pdat->nshape = n->attribute_read_ints("shape", &ints);
	if ((pdat->nshape > 0) && (pdat->shape = (int*)malloc(sizeof(int) * pdat->nshape))) {
		pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 1);
		pdat->high = n->attribute_read_float("high", 1.0);
		pdat->low = n->attribute_read_float("low", 0.0);
		pdat->seed = n->attribute_read_float("seed", 0.0);
		for (int i = 0; i < pdat->nshape; i++)
			pdat->shape[i] = ints[i];
		n->priv = pdat;
		return true;
	}else {
		delete pdat;
		return false;
	}
}

int RandomUniform_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;

	if (pdat) {
		if (pdat->shape)
			free(pdat->shape);
		delete pdat;
	}
	return 1;
}

int RandomUniform_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape(pdat->shape, pdat->nshape, pdat->dtype);
}

void RandomUniform_operator(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	if (pdat->seed != 0.0)
		srand(pdat->seed);
	switch (pdat->dtype) {
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (size_t i = 0, l = y->ndata; i < l; i++)
				py[i] = float16_to_float32(((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) + pdat->low);
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

void resolver_default_op_RandomUniform(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomUniform_operator;
	}
	if (n->ope) {
		n->init = RandomUniform_init;
		n->exit = RandomUniform_exit;
		n->reshape = RandomUniform_reshape;
	}
}
