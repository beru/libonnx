#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct RandomNormal_operator : public operator_t {

	tensor_type_t dtype;
	float mean;
	float scale;
	float seed;
	std::vector<int> shape;
	int nshape;

	bool init() override {
		if (n->outputs.size() != 1) {
			return false;
		}
		int64_t* ints;
		nshape = n->attribute("shape", &ints);
		if (nshape <= 0) {
			return false;
		}
		shape.resize(nshape);
		dtype = (tensor_type_t)n->attribute("dtype", ONNX_TENSOR_TYPE_FLOAT32);
		mean = n->attribute("mean", 0.0f);
		scale = n->attribute("scale", 1.0f);
		seed = n->attribute("seed", 0.0f);
		for (int i = 0; i < nshape; i++)
			shape[i] = ints[i];
		return true;
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		return y->reshape(&shape[0], nshape, dtype);
	}

	void exec() override {
		tensor_t* y = n->outputs[0];

		if (seed != 0.0)
			srand(seed);
		switch (dtype) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			{
				float16_t* py = (float16_t*)y->data;
				float ty, tx;
				for (size_t i = 0, l = y->ndata; i < l; i++) {
					ty = (float)rand() / (RAND_MAX + 1.0f);
					tx = (float)rand() / (RAND_MAX + 1.0f);
					py[i] = mean + scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty);
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
					py[i] = mean + scale * sqrtf(-2.0f * logf(tx)) * cosf(2.0f * acosf(-1.0f) * ty);
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
					py[i] = mean + scale * sqrt(-2.0f * log(tx)) * cos(2.0f * acos(-1.0f) * ty);
				}
			}
			break;
		default:
			break;
		}
	}

};

} // namespace {

void resolver_default_op_RandomNormal(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = new RandomNormal_operator;
	}
}

} // namespace onnx
