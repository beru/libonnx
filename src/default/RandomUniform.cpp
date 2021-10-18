#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void RandomUniform(T* py, size_t ndata, float high, float low)
{
	for (size_t i = 0; i < ndata; ++i) {
		py[i] = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
	}
}

struct RandomUniform_operator : public operator_t {
	tensor_type_t dtype;
	float high;
	float low;
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
		high = attribute("high", 1.0f);
		low = attribute("low", 0.0f);
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
			RandomUniform((float16_t*)y->data, y->ndata, high, low);
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			RandomUniform((float*)y->data, y->ndata, high, low);
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			RandomUniform((double*)y->data, y->ndata, high, low);
			break;
		default:
			break;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_RandomUniform(int opset)
{
	return new RandomUniform_operator;
}

} // namespace onnx
