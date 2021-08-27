#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	double start;
	double limit;
	double delta;
};

bool Range_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 3, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->start = 0;
	pdat->limit = 0;
	pdat->delta = 0;
	n->priv = pdat;
	return true;
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

int Range_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];

	pdat->start = tensor_get_value(n->inputs[0]->data, n->inputs[0]->type);
	pdat->limit = tensor_get_value(n->inputs[1]->data, n->inputs[1]->type);
	pdat->delta = tensor_get_value(n->inputs[2]->data, n->inputs[2]->type);
	int ndim = fmax(ceil((pdat->limit - pdat->start) / pdat->delta), 0);
	int tmp[] = { ndim };
	return y->reshape(tmp, 1, n->inputs[0]->type);
}

template <typename T>
void Range_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = pdat->start + (pdat->delta * i);
}

GEN_HOLEDR_TYPE(holder, Range_generic)

} // namespace

void resolver_default_op_Range(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_select<holder,
			int16_t, int32_t, int64_t,
			float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Range_init;
		n->reshape = Range_reshape;
	}
}
