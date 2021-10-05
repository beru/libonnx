#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Constant_operator : public operator_t {
	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		Onnx__AttributeProto* attr = proto->attribute[0];
		if (!attr) {
			return false;
		}
		tensor_t* y = outputs[0];
		std::string_view name(attr->name);
		switch (attr->type) {
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
			if (name == "value_float") {
				if ((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_FLOAT32))
					y->reinit(ONNX_TENSOR_TYPE_FLOAT32, nullptr, 0);
				y->apply(&attr->f, sizeof(float));
				return true;
			}
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
			if (name == "value_int") {
				if ((y->ndim != 0) || (y->type != ONNX_TENSOR_TYPE_INT64))
					y->reinit(ONNX_TENSOR_TYPE_INT64, nullptr, 0);
				y->apply(&attr->i, sizeof(int64_t));
				return true;
			}
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
			if ((name == "value_floats") && (attr->n_floats > 0)) {
				if ((y->ndim != 1) || (y->dims[0] != attr->n_floats) || (y->type != ONNX_TENSOR_TYPE_FLOAT32)) {
					int tmp[] = { (int)attr->n_floats };
					y->reinit(ONNX_TENSOR_TYPE_FLOAT32, tmp, 1);
				}
				y->apply(attr->floats, attr->n_floats * sizeof(float));
				return true;
			}
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
			if ((name == "value_ints") && (attr->n_ints > 0)) {
				if ((y->ndim != 1) || (y->dims[0] != attr->n_ints) || (y->type != ONNX_TENSOR_TYPE_INT64)) {
					int tmp[] = { (int)attr->n_ints };
					y->reinit(ONNX_TENSOR_TYPE_INT64, tmp, 1);
				}
				y->apply(attr->ints, attr->n_ints * sizeof(int64_t));
				return true;
			}
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
			if ((name == "value_strings") && (attr->n_strings > 0)) {
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
			if (attribute("value", y))
				return true;
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
			break;
		default:
			break;
		}
		return false;
	}

	void exec() override {
		if (opset >= 13) {
		}else if (opset >= 12) {
		}else if (opset >= 11) {
		}else if (opset >= 9) {
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Constant(int opset)
{
	return new Constant_operator;
}

} // namespace onnx
