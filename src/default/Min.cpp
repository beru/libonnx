#include <onnx.h>

static int Min_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Min_exit(onnx_node_t* n)
{
	return 1;
}

static int Min_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[0], n->inputs[0]->type))
		return 0;
	for (i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Min_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	T* py = (T*)y->datas;
	T* px;
	T minv;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, minv = std::numeric_limits<T>::max(); j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (T*)x->broadcast_map_address(y, i);
			if (*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

static void Min_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float v;
	float minv;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, minv = FLT_MAX; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			v = bfloat16_to_float32(*px);
			if (v < minv)
				minv = v;
		}
		py[i] = float32_to_bfloat16(minv);
	}
}

static void Min_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float v;
	float minv;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, minv = FLT_MAX; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			v = float16_to_float32(*px);
			if (v < minv)
				minv = v;
		}
		py[i] = float32_to_float16(minv);
	}
}

void resolver_default_op_Min(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Min_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Min_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Min_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Min_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Min_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Min_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Min_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Min_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Min_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Min_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Min_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 12) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Min_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Min_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Min_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Min_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Min_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Min_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Min_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Min_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Min_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Min_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 8) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Min_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Min_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 6) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Min_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Min_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Min_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Min_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Min_generic<double>;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Min_init;
		n->exit = Min_exit;
		n->reshape = Min_reshape;
	}
}
