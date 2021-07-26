#include <onnx.h>

static int Where_init(struct onnx_node_t * n)
{
	if((n->ninput == 3) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Where_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Where_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[n->ninput - 1], n->inputs[n->ninput - 1]->type))
		return 0;
	for(i = n->ninput - 2; i >= 0; i--)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Where_generic(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	T * py = (T *)y->datas;
	T * px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (T*)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (T*)onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (uint16_t*)onnx_tensor_broadcast_map_address(x2, y, i);
		py[i] = *px;
	}
}

static void Where_complex64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	float * py = (float *)y->datas;
	float * px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (float*)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (float*)onnx_tensor_broadcast_map_address(x2, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_complex128(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	double * py = (double *)y->datas;
	double * px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (double*)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (double*)onnx_tensor_broadcast_map_address(x2, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_string(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * x0 = n->inputs[0];
	struct onnx_tensor_t * x1 = n->inputs[1];
	struct onnx_tensor_t * x2 = n->inputs[2];
	char ** py = (char **)y->datas;
	char ** px;
	uint8_t * c;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		c = (uint8_t*)onnx_tensor_broadcast_map_address(x0, y, i);
		if(*c)
			px = (char **)onnx_tensor_broadcast_map_address(x1, y, i);
		else
			px = (char **)onnx_tensor_broadcast_map_address(x2, y, i);
		if(py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Where(struct onnx_node_t * n)
{
	if(n->opset >= 9)
	{
		if(n->ninput == 3)
		{
			switch(n->inputs[2]->type)
			{
			case ONNX_TENSOR_TYPE_BOOL:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<uint8_t>;
				break;
			case ONNX_TENSOR_TYPE_INT8:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<int8_t>;
				break;
			case ONNX_TENSOR_TYPE_INT16:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<int16_t>;
				break;
			case ONNX_TENSOR_TYPE_INT32:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<int32_t>;
				break;
			case ONNX_TENSOR_TYPE_INT64:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<int64_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT8:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<uint8_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT16:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<uint16_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT32:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<uint32_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT64:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<uint64_t>;
				break;
			case ONNX_TENSOR_TYPE_BFLOAT16:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_bfloat16;
				break;
			case ONNX_TENSOR_TYPE_FLOAT16:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_float16;
				break;
			case ONNX_TENSOR_TYPE_FLOAT32:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<float>;
				break;
			case ONNX_TENSOR_TYPE_FLOAT64:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_generic<double>;
				break;
			case ONNX_TENSOR_TYPE_COMPLEX64:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_complex64;
				break;
			case ONNX_TENSOR_TYPE_COMPLEX128:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_complex128;
				break;
			case ONNX_TENSOR_TYPE_STRING:
				n->init = Where_init;
				n->exit = Where_exit;
				n->reshape = Where_reshape;
				n->ope = Where_string;
				break;
			default:
				break;
			}
		}
	}
}
