#include <onnx.h>

static int Mul_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Mul_exit(onnx_node_t* n)
{
	return 1;
}

static int Mul_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
static void Mul_generic(onnx_node_t* n)
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
		py[i] = *pa * *pb;
	}
}

static void Mul_bfloat16(onnx_node_t* n)
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
		py[i] = float32_to_bfloat16(bfloat16_to_float32(*pa) * bfloat16_to_float32(*pb));
	}
}

static void Mul_float16(onnx_node_t* n)
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
		py[i] = float32_to_float16(float16_to_float32(*pa) * float16_to_float32(*pb));
	}
}

void resolver_default_op_Mul(onnx_node_t* n)
{
	if (n->opset >= 14) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Mul_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Mul_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Mul_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Mul_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Mul_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Mul_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Mul_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Mul_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Mul_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Mul_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Mul_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Mul_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Mul_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Mul_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Mul_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Mul_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Mul_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Mul_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Mul_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Mul_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 7) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Mul_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Mul_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Mul_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Mul_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Mul_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Mul_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Mul_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Mul_init;
		n->exit = Mul_exit;
		n->reshape = Mul_reshape;
	}
}
