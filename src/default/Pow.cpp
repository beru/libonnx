#include <onnx.h>

static int Pow_init(onnx_node_t * n)
{
	if((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Pow_exit(onnx_node_t * n)
{
	return 1;
}

static int Pow_reshape(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

static double tensor_get_value(void * p, onnx_tensor_type_t type)
{
	double v;

	switch(type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
		v = *((uint8_t *)p);
		break;
	case ONNX_TENSOR_TYPE_INT8:
		v = *((int8_t *)p);
		break;
	case ONNX_TENSOR_TYPE_INT16:
		v = *((int16_t *)p);
		break;
	case ONNX_TENSOR_TYPE_INT32:
		v = *((int32_t *)p);
		break;
	case ONNX_TENSOR_TYPE_INT64:
		v = *((int64_t *)p);
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		v = *((uint8_t *)p);
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		v = *((uint16_t *)p);
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		v = *((uint32_t *)p);
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		v = *((uint64_t *)p);
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		v = bfloat16_to_float32(*((uint16_t *)p));
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		v = float16_to_float32(*((uint16_t *)p));
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		v = *((float *)p);
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		v = *((double *)p);
		break;
	default:
		v = 0;
		break;
	}
	return v;
}

template <typename T>
static void Pow_generic(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	T * py = (T *)y->datas;
	T * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

static void Pow_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t*)onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = float32_to_bfloat16(pow(bfloat16_to_float32(*pa), v));
	}
}

static void Pow_float16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t*)onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = float32_to_float16(pow(float16_to_float32(*pa), v));
	}
}

void resolver_default_op_Pow(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<double>;
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
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->ope = Pow_generic<double>;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
