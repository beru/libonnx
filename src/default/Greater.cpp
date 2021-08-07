#include <onnx.h>

static int Greater_init(onnx_node_t * n)
{
	if((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Greater_exit(onnx_node_t * n)
{
	return 1;
}

static int Greater_reshape(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
static void Greater_generic(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	T * pa;
	T * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t *)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t *)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (bfloat16_to_float32(*pa) > bfloat16_to_float32(*pb)) ? 1 : 0;
	}
}

static void Greater_float16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t *)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t *)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (float16_to_float32(*pa) > float16_to_float32(*pb)) ? 1 : 0;
	}
}

void resolver_default_op_Greater(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->ope = Greater_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
