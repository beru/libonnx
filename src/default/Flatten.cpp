#include <onnx.h>
#include "util.h"

namespace onnx {

struct ope_pdata_t : public node_t::ope_pdata_t {
	int axis;
};

bool Flatten_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	pdat->axis = n->attribute("axis", 1);
	n->priv = pdat;
	return true;
}

int Flatten_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int axis = pdat->axis;
	std::vector<int> dims(x->ndim);
	int ndim;
	int i, j;

	if (axis < 0)
		axis += x->ndim;
	if (axis < 0 || axis >= x->ndim)
		return 0;
	for (i = 0, j = 1, ndim = 0; i < x->ndim; i++) {
		if (i != axis)
			j *= x->dims[i];
		else {
			dims[ndim++] = j;
			j = x->dims[i];
		}
	}
	dims[ndim++] = j;
	return y->reshape(&dims[0], ndim, x->type);
}

void Flatten_ope(node_t* n)
{
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	if (x->type == ONNX_TENSOR_TYPE_STRING) {
		const std::string* px = (const std::string*)x->data;
		std::string* py = (std::string*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			py[i] = px[i];
		}
	}else {
		memcpy(y->data, x->data, x->ndata * tensor_type_sizeof(x));
	}
}

void resolver_default_op_Flatten(node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_BFLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->ope = Flatten_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->ope = Flatten_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 9) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->ope = Flatten_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Flatten_ope;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Flatten_init;
		n->reshape = Flatten_reshape;
	}
}

} // namespace onnx
