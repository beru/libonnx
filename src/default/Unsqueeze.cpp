#include <onnx.h>
#include "util.h"
	
namespace onnx {

namespace {

int Unsqueeze_reshape(node_t* n)
{
	tensor_t* y = n->outputs[0];
	tensor_t* x = n->inputs[0];
	tensor_t* a = n->inputs[1];
	int64_t* pa = (int64_t*)a->data;
	int ndim = x->ndim + a->ndata;
	std::vector<int> dims(ndim);
	int i, j;

	memset(&dims[0], 0, sizeof(int) * ndim);
	for (i = 0; i < a->ndata; i++) {
		int axis = pa[i];
		if (axis < 0)
			axis += ndim;
		if (axis >= 0 && axis < ndim)
			dims[axis] = 1;
	}
	for (i = 0, j = 0; i < ndim; i++) {
		if (dims[i] != 1)
			dims[i] = x->dims[j++];
	}
	return y->reshape(&dims[0], ndim, x->type);
}

void Unsqueeze_ope(node_t* n)
{
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	if (x->type == ONNX_TENSOR_TYPE_STRING) {
		std::string* px = (std::string*)x->data;
		std::string* py = (std::string*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			py[i] = px[i];
		}
	}else {
		memcpy(y->data, x->data, x->ndata * tensor_type_sizeof(x));
	}
}

} // namespace

void resolver_default_op_Unsqueeze(node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
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
			n->ope = Unsqueeze_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = [](node_t* n){
			return is_inout_size(n, 2, 1);
		};
		n->reshape = Unsqueeze_reshape;
	}
}

} // namespace onnx
