#include <onnx.h>

static int Abs_init(onnx_node_t * n)
{
	if((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Abs_exit(onnx_node_t * n)
{
	return 1;
}

static int Abs_reshape(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

template <typename T>
static void Abs_generic(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	T * px = (T *)x->datas;
	T * py = (T *)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if constexpr (std::is_signed_v<T>) {
			py[i] = abs(px[i]);
		}else {
			py[i] = px[i];
		}
	}
}

static void Abs_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(fabsf(v));
	}
}

static void Abs_float16(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(fabsf(v));
	}
}

void resolver_default_op_Abs(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Abs_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Abs_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Abs_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Abs_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Abs_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Abs_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Abs_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Abs_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = Abs_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Abs_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Abs_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Abs_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = Abs_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->ope = Abs_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = Abs_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = Abs_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = Abs_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->ope = Abs_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = Abs_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = Abs_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Abs_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Abs_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Abs_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Abs_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Abs_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Abs_generic<double>;
			break;
		default:
			break;
		}
	}

	if (n->ope) {
		n->init = Abs_init;
		n->exit = Abs_exit;
		n->reshape = Abs_reshape;
	}
}
