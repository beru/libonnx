#include "onnx.h"
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
		if (outputs.size() != 1) {
			return false;
		}
		int64_t* ints;
		nshape = attribute("shape", ints);
		if (nshape <= 0) {
			return false;
		}
		shape.resize(nshape);
		dtype = (tensor_type_t)attribute("dtype", ONNX_TENSOR_TYPE_FLOAT32);
		mean = attribute("mean", 0.0f);
		scale = attribute("scale", 1.0f);
		seed = attribute("seed", 0.0f);
		for (int i = 0; i < nshape; ++i) {
			shape[i] = ints[i];
		}
		return true;
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		return y->reshape(&shape[0], nshape, dtype);
	}

	void exec() override {
		if (opset < 1) {
			return;
		}
		tensor_t* y = outputs[0];

		if (seed != 0.0) {
			srand((unsigned int)seed);
		}
		switch (dtype) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			{
				float16_t* py = (float16_t*)y->data;
				for (size_t i = 0, l = y->ndata; i < l; ++i) {
					float ty = (float)rand() / (RAND_MAX + 1.0f);
					float tx = (float)rand() / (RAND_MAX + 1.0f);
					py[i] = mean + scale * sqrt(-2.0f * log(tx)) * cos(2.0f * acos(-1.0f) * ty);
				}
			}
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			{
				float* py = (float*)y->data;
				for (size_t i = 0, l = y->ndata; i < l; ++i) {
					float ty = (float)rand() / (RAND_MAX + 1.0f);
					float tx = (float)rand() / (RAND_MAX + 1.0f);
					py[i] = mean + scale * sqrt(-2.0f * log(tx)) * cos(2.0f * acos(-1.0f) * ty);
				}
			}
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			{
				double* py = (double*)y->data;
				for (size_t i = 0, l = y->ndata; i < l; ++i) {
					double ty = (double)rand() / (RAND_MAX + 1.0);
					double tx = (double)rand() / (RAND_MAX + 1.0);
					py[i] = mean + scale * sqrt(-2.0 * log(tx)) * cos(2.0 * acos(-1.0) * ty);
				}
			}
			break;
		default:
			break;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_RandomNormal(int opset) { return new (std::nothrow) RandomNormal_operator; }

} // namespace onnx
