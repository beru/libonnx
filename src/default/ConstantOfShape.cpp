#include <onnx.h>
#include "util.h"

namespace {

union onnx_scalar_t {
	uint8_t v_bool;
	int8_t v_int8;
	int16_t v_int16;
	int32_t v_int32;
	int64_t v_int64;
	uint8_t v_uint8;
	uint16_t v_uint16;
	uint32_t v_uint32;
	uint64_t v_uint64;
	uint16_t v_bfloat16;
	uint16_t v_float16;
	float v_float32;
	double v_float64;
	struct {
		float real;
		float imaginary;
	} v_complex64;
	struct {
		double real;
		double imaginary;
	} v_complex128;
};

struct ope_pdata_t : public onnx_node_t::ope_pdata_t {
	onnx_tensor_type_t type;
	onnx_scalar_t scalar;
	int size;
};

bool ConstantOfShape_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	Onnx__TensorProto* t = nullptr;
	for (int i = 0; i < n->proto->n_attribute; i++) {
		Onnx__AttributeProto* attr = n->proto->attribute[i];
		if ((attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) && (strcmp(attr->name, "value") == 0)) {
			t = attr->t;
			break;
		}
	}
	if (t) {
		pdat->type = (onnx_tensor_type_t)t->data_type;
		switch (t->data_type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			pdat->scalar.v_float32 = t->float_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			pdat->scalar.v_uint8 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			pdat->scalar.v_int8 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			pdat->scalar.v_uint16 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			pdat->scalar.v_int16 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			pdat->scalar.v_int32 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			pdat->scalar.v_bool = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			pdat->scalar.v_float16 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			pdat->scalar.v_bfloat16 = t->int32_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			pdat->scalar.v_int64 = t->int64_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			pdat->scalar.v_float64 = t->double_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			pdat->scalar.v_uint32 = t->uint64_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			pdat->scalar.v_uint64 = t->uint64_data[0];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			pdat->scalar.v_complex64.real = t->float_data[0];
			pdat->scalar.v_complex64.imaginary = t->float_data[1];
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			pdat->scalar.v_complex128.real = t->double_data[0];
			pdat->scalar.v_complex128.imaginary = t->double_data[1];
			break;
		default:
			memset(&pdat->scalar, 0, sizeof(onnx_scalar_t));
			break;
		}
	}else {
		pdat->type = ONNX_TENSOR_TYPE_FLOAT32;
		memset(&pdat->scalar, 0, sizeof(onnx_scalar_t));
	}
	pdat->size = onnx_tensor_type_sizeof(pdat->type);
	n->priv = pdat;
	return true;
}

int ConstantOfShape_reshape(onnx_node_t* n)
{
	return 1;
}

void ConstantOfShape_ope(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	char* p;
	size_t i, l;

	if (x->ndata > 0) {
		std::vector<int> dims(x->ndata);
		for (i = 0; i < x->ndata; i++)
			dims[i] = ((int64_t*)x->data)[i];
		y->reinit(pdat->type, &dims[0], x->ndata);
	}else {
		y->reinit(pdat->type, nullptr, 0);
	}
	for (i = 0, l = y->ndata, p = (char*)y->data; i < l; i++, p += pdat->size)
		memcpy(p, &pdat->scalar, pdat->size);
}

} // namespace

void resolver_default_op_ConstantOfShape(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->init = ConstantOfShape_init;
		n->reshape = ConstantOfShape_reshape;
		n->ope = ConstantOfShape_ope;
	}
}
