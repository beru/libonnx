#include <onnx.h>

static int GreaterOrEqual_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int GreaterOrEqual_exit(struct onnx_node_t * n)
{
	return 1;
}

static int GreaterOrEqual_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
static void GreaterOrEqual_generic(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	T * pa;
	T * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa >= *pb) ? 1 : 0;
	}
}

static void GreaterOrEqual_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t *)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t *)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (float16_to_float32(*pa) >= float16_to_float32(*pb)) ? 1 : 0;
	}
}

void resolver_default_op_GreaterOrEqual(struct onnx_node_t * n)
{
	if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = GreaterOrEqual_init;
			n->exit = GreaterOrEqual_exit;
			n->reshape = GreaterOrEqual_reshape;
			n->ope = GreaterOrEqual_generic<double>;
			break;
		default:
			break;
		}
	}
}
