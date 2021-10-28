#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct RandomNormalLike_operator : public operator_t {
	tensor_type_t dtype;
	float mean;
	float scale;
	float seed;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		dtype = (tensor_type_t)attribute("dtype", ONNX_TENSOR_TYPE_UNDEFINED);
		mean = attribute("mean", 0.0f);
		scale = attribute("scale", 1.0f);
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

operator_t* resolver_default_op_RandomNormalLike(int opset) { return new (std::nothrow) RandomNormalLike_operator; }

} // namespace onnx
