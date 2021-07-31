#include <onnx.h>

static int Tile_init(onnx_node_t * n)
{
	if((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Tile_exit(onnx_node_t * n)
{
	return 1;
}

static int Tile_reshape(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * r = n->inputs[1];
	int64_t * pr = (int64_t *)r->datas;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	for(i = 0; i < ndim; i++)
		dims[i] = x->dims[i] * pr[i];
	return onnx_tensor_reshape(y, &dims[0], ndim, x->type);
}

template <typename T>
static void Tile_generic(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	T * py = (T *)y->datas;
	T * px = (T *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (T*)onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (uint16_t *)onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_float16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px = (uint16_t *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (uint16_t *)onnx_tensor_broadcast_map_address(x, y, i);
		py[i] = *px;
	}
}

static void Tile_complex64(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	float * py = (float *)y->datas;
	float * px = (float *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (float*)onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Tile_complex128(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	double * py = (double *)y->datas;
	double * px = (double *)x->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (double*)onnx_tensor_broadcast_map_address(x, y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Tile_string(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		px = (char**)onnx_tensor_broadcast_map_address(x, y, i);
		if(py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Tile(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<double>;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_generic<double>;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Tile_init;
			n->exit = Tile_exit;
			n->reshape = Tile_reshape;
			n->ope = Tile_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
