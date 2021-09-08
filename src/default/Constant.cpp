#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

bool Constant_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	Onnx__AttributeProto* attr = n->proto->attribute[0];
	if (!attr) {
		return false;
	}
	tensor_t* y = n->outputs[0];
	switch (attr->type) {
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
		if (strcmp(attr->name, "value_float") == 0) {
			if ((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_FLOAT32))
				y->reinit(ONNX_TENSOR_TYPE_FLOAT32, nullptr, 0);
			y->apply(&attr->f, sizeof(float));
			return true;
		}
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
		if (strcmp(attr->name, "value_int") == 0) {
			if ((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_INT64))
				y->reinit(ONNX_TENSOR_TYPE_INT64, nullptr, 0);
			y->apply(&attr->i, sizeof(int64_t));
			return true;
		}
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
		if ((strcmp(attr->name, "value_floats") == 0) && (attr->n_floats > 0)) {
			if ((y->ndim != 1) || (y->dims[0] != attr->n_floats) || (y->type != ONNX_TENSOR_TYPE_FLOAT32)) {
				int tmp[] = { (int)attr->n_floats };
				y->reinit(ONNX_TENSOR_TYPE_FLOAT32, tmp, 1);
			}
			y->apply(attr->floats, attr->n_floats * sizeof(float));
			return true;
		}
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
		if ((strcmp(attr->name, "value_ints") == 0) && (attr->n_ints > 0)) {
			if ((y->ndim != 1) || (y->dims[0] != attr->n_ints) || (y->type != ONNX_TENSOR_TYPE_INT64)) {
				int tmp[] = { (int)attr->n_ints };
				y->reinit(ONNX_TENSOR_TYPE_INT64, tmp, 1);
			}
			y->apply(attr->ints, attr->n_ints * sizeof(int64_t));
			return true;
		}
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
		if ((strcmp(attr->name, "value_strings") == 0) && (attr->n_strings > 0)) {
			if ((y->ndim != 1) || (y->dims[0] != attr->n_strings) || (y->type != ONNX_TENSOR_TYPE_STRING)) {
				int tmp[] = { (int)attr->n_ints };
				y->reinit(ONNX_TENSOR_TYPE_STRING, tmp, 1);
			}
			if (y->data && attr->strings) {
				std::string* str = (std::string*)y->data;
				for (size_t i = 0; i < y->ndata; i++) {
					str[i].assign((const char*)attr->strings[i].data, attr->strings[i].len);
				}
			}
			return true;
		}
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
		if (n->read_attribute("value", y))
			return true;
		break;
	case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
		break;
	default:
		break;
	}
	return false;
}

int Constant_reshape(node_t* n)
{
	return 1;
}

void Constant_ope(node_t* n)
{
}

} // namespace

void resolver_default_op_Constant(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = Constant_ope;
	}else if (n->opset >= 12) {
		n->ope = Constant_ope;
	}else if (n->opset >= 11) {
		n->ope = Constant_ope;
	}else if (n->opset >= 9) {
		n->ope = Constant_ope;
	}else if (n->opset >= 1) {
		n->ope = Constant_ope;
	}
	if (n->ope) {
		n->init = Constant_init;
		n->reshape = Constant_reshape;
	}
}

} // namespace onnx
