#include <onnx.h>
#include "util.h"

namespace onnx {

static
double tensor_get_value(void* p, tensor_type_t type)
{
	double v;

	switch (type) {
	case ONNX_TENSOR_TYPE_BOOL:
		v = *((bool_t*)p);
		break;
	case ONNX_TENSOR_TYPE_INT8:
		v = *((int8_t*)p);
		break;
	case ONNX_TENSOR_TYPE_INT16:
		v = *((int16_t*)p);
		break;
	case ONNX_TENSOR_TYPE_INT32:
		v = *((int32_t*)p);
		break;
	case ONNX_TENSOR_TYPE_INT64:
		v = *((int64_t*)p);
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		v = *((uint8_t*)p);
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		v = *((uint16_t*)p);
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		v = *((uint32_t*)p);
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		v = *((uint64_t*)p);
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		v = *((bfloat16_t*)p);
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		v = *((float16_t*)p);
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		v = *((float*)p);
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		v = *((double*)p);
		break;
	default:
		v = 0;
		break;
	}
	return v;
}

template <typename T>
struct Range_operator : public operator_t {
	double start = 0;
	double limit = 0;
	double delta = 0;

	bool init() override {
		if (!is_inout_size(n, 3, 1)) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		start = tensor_get_value(n->inputs[0]->data, n->inputs[0]->type);
		limit = tensor_get_value(n->inputs[1]->data, n->inputs[1]->type);
		delta = tensor_get_value(n->inputs[2]->data, n->inputs[2]->type);
		int ndim = fmax(ceil((limit - start) / delta), 0);
		int tmp[] = { ndim };
		return y->reshape(tmp, 1, n->inputs[0]->type);
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++)
			py[i] = start + (delta * i);
	}
};

void resolver_default_op_Range(node_t* n)
{
	if (n->opset >= 11) {
		n->ope = ope_type_select<Range_operator,
			int16_t, int32_t, int64_t,
			float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
