#include <onnx.h>

static int Shape_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Shape_exit(onnx_node_t* n)
{
	return 1;
}

static int Shape_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	int tmp[] = { x->ndim };
	return y->reshape(tmp, 1, ONNX_TENSOR_TYPE_INT64);
}

static void Shape_ope(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t* py = (int64_t*)y->datas;
	size_t i, l;

	for (i = 0, l = min(y->ndata, (size_t)x->ndim); i < l; i++)
		py[i] = x->dims[i];
}

void resolver_default_op_Shape(onnx_node_t* n)
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
			n->init = Shape_init;
			n->exit = Shape_exit;
			n->reshape = Shape_reshape;
			n->ope = Shape_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
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
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Shape_init;
			n->exit = Shape_exit;
			n->reshape = Shape_reshape;
			n->ope = Shape_ope;
			break;
		default:
			break;
		}
	}
}
