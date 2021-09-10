#include <onnx.h>
#include "util.h"

namespace onnx {

struct RandomNormalLike_operator : public operator_t {
	tensor_type_t dtype;
	float mean;
	float scale;
	float seed;

	bool init() override {
		if (!is_inout_size(n, 1, 1)) {
			return false;
		}
		dtype = (tensor_type_t)n->attribute("dtype", ONNX_TENSOR_TYPE_UNDEFINED);
		mean = n->attribute("mean", 0.0f);
		scale = n->attribute("scale", 1.0f);
		seed = n->attribute("seed", 0.0f);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		tensor_type_t type;

		if (dtype != ONNX_TENSOR_TYPE_UNDEFINED)
			type = dtype;
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

void resolver_default_op_RandomNormalLike(node_t* n)
{
	if (n->opset >= 1) {
		n->ope = std::make_shared<RandomNormalLike_operator>();
	}
}

} // namespace onnx
