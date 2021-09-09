#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	~operator_pdata_t() {
	}
	tensor_type_t dtype;
	float high;
	float low;
	float seed;
	std::vector<int> shape;
	int nshape;
};

bool RandomUniform_init(node_t* n)
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
	pdat->high = n->attribute("high", 1.0f);
	pdat->low = n->attribute("low", 0.0f);
	pdat->seed = n->attribute("seed", 0.0f);
	for (int i = 0; i < pdat->nshape; i++)
		pdat->shape[i] = ints[i];
	n->priv = pdat;
	return true;
}

int RandomUniform_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];

	return y->reshape(&pdat->shape[0], pdat->nshape, pdat->dtype);
}

template <typename T>
void RandomUniform(T* py, size_t ndata, float high, float low)
{
	for (size_t i = 0; i < ndata; i++)
		py[i] = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
}

void RandomUniform_operator(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];

	if (pdat->seed != 0.0)
		srand(pdat->seed);
	switch (pdat->dtype) {
	case ONNX_TENSOR_TYPE_FLOAT16:
		RandomUniform((float16_t*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		RandomUniform((float*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		RandomUniform((double*)y->data, y->ndata, pdat->high, pdat->low);
		break;
	default:
		break;
	}
}

} // namespace

void resolver_default_op_RandomUniform(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = RandomUniform_operator;
	}
	if (n->ope) {
		n->init = RandomUniform_init;
		n->reshape = RandomUniform_reshape;
	}
}

} // namespace onnx
