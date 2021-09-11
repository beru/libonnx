#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

union scalar_t {
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

struct ConstantOfShape_operator : public operator_t {
	tensor_type_t type;
	scalar_t scalar;
	int size;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		Onnx__TensorProto* t = nullptr;
		for (int i = 0; i < n->proto->n_attribute; i++) {
			Onnx__AttributeProto* attr = n->proto->attribute[i];
			if ((attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) && (strcmp(attr->name, "value") == 0)) {
				t = attr->t;
				break;
			}
		}
		if (t) {
			type = (tensor_type_t)t->data_type;
			switch (t->data_type) {
			case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
				scalar.v_float32 = t->float_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
				scalar.v_uint8 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
				scalar.v_int8 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
				scalar.v_uint16 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
				scalar.v_int16 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
				scalar.v_int32 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
				scalar.v_bool = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
				scalar.v_float16 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
				scalar.v_bfloat16 = t->int32_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
				scalar.v_int64 = t->int64_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
				scalar.v_float64 = t->double_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
				scalar.v_uint32 = t->uint64_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
				scalar.v_uint64 = t->uint64_data[0];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
				scalar.v_complex64.real = t->float_data[0];
				scalar.v_complex64.imaginary = t->float_data[1];
				break;
			case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
				scalar.v_complex128.real = t->double_data[0];
				scalar.v_complex128.imaginary = t->double_data[1];
				break;
			default:
				memset(&scalar, 0, sizeof(scalar_t));
				break;
			}
		}else {
			type = ONNX_TENSOR_TYPE_FLOAT32;
			memset(&scalar, 0, sizeof(scalar_t));
		}
		size = tensor_type_sizeof(type);
		return true;
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		char* p;
		size_t i, l;

		if (x->ndata > 0) {
			std::vector<int> dims(x->ndata);
			for (i = 0; i < x->ndata; i++)
				dims[i] = ((int64_t*)x->data)[i];
			y->reinit(type, &dims[0], x->ndata);
		}else {
			y->reinit(type, nullptr, 0);
		}
		for (i = 0, l = y->ndata, p = (char*)y->data; i < l; i++, p += size)
			memcpy(p, &scalar, size);
	}

};

} // namespace {

void resolver_default_op_ConstantOfShape(node_t* n)
{
	if (n->opset >= 9) {
		n->ope = std::make_shared<ConstantOfShape_operator>();
	}
}

} // namespace onnx
