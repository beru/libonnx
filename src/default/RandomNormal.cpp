#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	~operator_pdata_t() {
		if (shape)
			free(shape);
	}
	onnx_tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
	int* shape = nullptr;
	int nshape;
};

bool RandomNormal_init(onnx_node_t* n)
{
	if (n->outputs.size() != 1) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	int64_t* ints;
	pdat->nshape = n->read_attribute("shape", &ints);
	if ((pdat->nshape > 0) && (pdat->shape = (int*)malloc(sizeof(int) * pdat->nshape))) {
		pdat->dtype = (onnx_tensor_type_t)n->read_attribute("dtype", ONNX_TENSOR_TYPE_FLOAT32);
		pdat->mean = n->read_attribute("mean", 0.0f);
		pdat->scale = n->read_attribute("scale", 1.0f);
		pdat->seed = n->read_attribute("seed", 0.0f);
		for (int i = 0; i < pdat->nshape; i++)
			pdat->shape[i] = ints[i];
		n->priv = pdat;
		return true;
	}else {
		delete pdat;
		return false;
	}
}

int RandomNormal_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape(pdat->shape, pdat->nshape, pdat->dtype);
}

void RandomNormal_operator(onnx_node_t* n)
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

void resolver_default_op_RandomNormal(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomNormal_operator;
	}
	if (n->ope) {
		n->init = RandomNormal_init;
		n->reshape = RandomNormal_reshape;
	}
}
