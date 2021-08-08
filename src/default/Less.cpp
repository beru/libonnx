#include <onnx.h>

static int Less_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Less_exit(onnx_node_t* n)
{
	return 1;
}

static int Less_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
static void Less_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	T* pa;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (T*)a->broadcast_map_address(y, i);
		pb = (T*)b->broadcast_map_address(y, i);
		py[i] = (*pa < *pb) ? 1 : 0;
	}
}

static void Less_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = (bfloat16_to_float32(*pa) < bfloat16_to_float32(*pb)) ? 1 : 0;
	}
}

static void Less_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = (float16_to_float32(*pa) < float16_to_float32(*pb)) ? 1 : 0;
	}
}

void resolver_default_op_Less(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 9) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 7) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Less_init;
			n->exit = Less_exit;
			n->reshape = Less_reshape;
			n->ope = Less_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
	}
}
