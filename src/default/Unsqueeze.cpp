#include <onnx.h>
#include "util.h"
	
namespace {

bool Unsqueeze_init(onnx_node_t* n)
{
	return is_inout_size(n, 2, 1);
}

int Unsqueeze_exit(onnx_node_t* n)
{
	return 1;
}

int Unsqueeze_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* a = n->inputs[1];
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

void Unsqueeze_ope(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	char** px = (char**)x->data;
	char** py = (char**)y->data;

	if (x->type == ONNX_TENSOR_TYPE_STRING) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (py[i])
				free(py[i]);
			py[i] = strdup(px[i]);
		}
	}else {
		memcpy(y->data, x->data, x->ndata * onnx_tensor_type_sizeof(x));
	}
}

} // namespace

void resolver_default_op_Unsqueeze(onnx_node_t* n)
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
		n->init = Unsqueeze_init;
		n->exit = Unsqueeze_exit;
		n->reshape = Unsqueeze_reshape;
	}
}
