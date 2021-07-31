#include <onnx.h>

static int Max_init(onnx_node_t * n)
{
	if((n->inputs.size() >= 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Max_exit(onnx_node_t * n)
{
	return 1;
}

static int Max_reshape(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[0], n->inputs[0]->type))
		return 0;
	for(i = 1; i < n->inputs.size(); i++)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Max_generic(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x;
	T * py = (T *)y->datas;
	T * px;
	T maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = std::numeric_limits<T>::min();
		for(int j = 0; j < n->inputs.size(); j++)
		{
			x = n->inputs[j];
			px = (T*)onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = FLT_MIN;
		for(int j = 0; j < n->inputs.size(); j++)
		{
			x = n->inputs[j];
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x, y, i);
			v = bfloat16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_bfloat16(maxv);
	}
}

static void Max_float16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = FLT_MIN;
		for(int j = 0; j < n->inputs.size(); j++)
		{
			x = n->inputs[j];
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x, y, i);
			v = float16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_float16(maxv);
	}
}

void resolver_default_op_Max(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 8)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<double>;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->ope = Max_generic<double>;
			break;
		default:
			break;
		}
	}
}
