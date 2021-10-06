#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void RandomUniformLike(T* py, size_t ndata, float high, float low)
{
	for (size_t i = 0; i < ndata; i++) {
		py[i] = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
	}
}

struct RandomUniformLike_operator : public operator_t {
	tensor_type_t dtype;
	float high;
	float low;
	float seed;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		dtype = (tensor_type_t)attribute("dtype", ONNX_TENSOR_TYPE_UNDEFINED);
		high = attribute("high", 1.0f);
		low = attribute("low", 0.0f);
		seed = attribute("seed", 0.0f);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		tensor_type_t type;

		if (dtype != ONNX_TENSOR_TYPE_UNDEFINED) {
			type = dtype;
		}else {
			type = x->type;
		}
		switch (type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
			return y->reshape(&x->dims[0], x->ndim, type);
		default:
			break;
		}
		return false;
	}

	void exec() override {
		if (opset < 1) {
			return;
		}

		tensor_t* y = outputs[0];
		if (seed != 0.0) {
			srand(seed);
		}
		switch (dtype) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			RandomUniformLike((float16_t*)y->data, y->ndata, high, low);
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			RandomUniformLike((float*)y->data, y->ndata, high, low);
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			RandomUniformLike((double*)y->data, y->ndata, high, low);
			break;
		default:
			break;
		}
	}
};

} // namespace {

operator_t* resolver_default_op_RandomUniformLike(int opset)
{
	return new RandomUniformLike_operator;
}

} // namespace onnx
