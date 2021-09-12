#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

double tensor_get_value(const void* p, tensor_type_t type)
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

struct Pow_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}
	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		return y->reshape_multi_broadcast(a, b, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			const T* pa = (const T*)a->broadcast_map_address(y, i);
			const void* pb = b->broadcast_map_address(y, i);
			T v = tensor_get_value(pb, b->type);
			py[i] = pow(*pa, v);
		}
	}

	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Pow_operator,
				int32_t, int64_t,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 12) {
			typed_exec<Pow_operator,
				int32_t, int64_t,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 7) {
			typed_exec<Pow_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
		}
	}

};

} // namespace {

void resolver_default_op_Pow(node_t* n)
{
	n->ope = std::make_shared<Pow_operator>();
}

} // namespace onnx
