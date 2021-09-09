#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	~operator_pdata_t() {
	}
	tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
	std::vector<int> shape;
	int nshape;
};

bool RandomNormal_init(node_t* n)
{
	if (n->outputs.size() != 1) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	int64_t* ints;
	pdat->nshape = n->attribute("shape", &ints);
	if (pdat->nshape <= 0) {
		return false;
	}
	pdat->shape.resize(pdat->nshape);
	pdat->dtype = (tensor_type_t)n->attribute("dtype", ONNX_TENSOR_TYPE_FLOAT32);
	pdat->mean = n->attribute("mean", 0.0f);
	pdat->scale = n->attribute("scale", 1.0f);
	pdat->seed = n->attribute("seed", 0.0f);
	for (int i = 0; i < pdat->nshape; i++)
		pdat->shape[i] = ints[i];
	n->priv = pdat;
	return true;
}

int RandomNormal_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];

	return y->reshape(&pdat->shape[0], pdat->nshape, pdat->dtype);
}

void RandomNormal_operator(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
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

void resolver_default_op_RandomNormal(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomNormal_operator;
	}
	if (n->ope) {
		n->init = RandomNormal_init;
		n->reshape = RandomNormal_reshape;
	}
}

} // namespace onnx
