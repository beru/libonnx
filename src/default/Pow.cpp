#include <onnx.h>
#include "util.h"

namespace {

bool Pow_init(onnx_node_t* n)
{
	return is_inout_size(n, 2, 1);
}

int Pow_exit(onnx_node_t* n)
{
	return 1;
}

int Pow_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

double tensor_get_value(void* p, onnx_tensor_type_t type)
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
void Pow_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T* pa = (T*)a->broadcast_map_address(y, i);
		void* pb = b->broadcast_map_address(y, i);
		T v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

GEN_HOLEDR_TYPE(holder, Pow_generic)

} // namespace

void resolver_default_op_Pow(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			int32_t, int64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_select<holder,
			int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Pow_init;
		n->exit = Pow_exit;
		n->reshape = Pow_reshape;
	}
}
