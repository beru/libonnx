#include <onnx.h>

static int Div_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Div_exit(onnx_node_t* n)
{
	return 1;
}

static int Div_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
static void Div_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (T*)a->broadcast_map_address(y, i);
		pb = (T*)b->broadcast_map_address(y, i);
		py[i] = *pa / *pb;
	}
}

static void Div_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = float32_to_float16(float16_to_float32(*pa) / float16_to_float32(*pb));
	}
}

static void Div_13_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) / bfloat16_to_float32(*pb));
	}
}

void resolver_default_op_Div(onnx_node_t* n)
{
	if (n->opset >= 14) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Div_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Div_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Div_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Div_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Div_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Div_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Div_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Div_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Div_13_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Div_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Div_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Div_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 13) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Div_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Div_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Div_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Div_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Div_13_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Div_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Div_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Div_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 7) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Div_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Div_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Div_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Div_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Div_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Div_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Div_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Div_init;
		n->exit = Div_exit;
		n->reshape = Div_reshape;
	}
}
