#include <onnx.h>

struct operator_pdata_t {
	int fmod;
};

static int Mod_init(onnx_node_t * n)
{
	operator_pdata_t * pdat;

	if((n->inputs.size() == 2) && (n->outputs.size() == 1))
	{
		pdat = (operator_pdata_t *)malloc(sizeof(operator_pdata_t));
		if(pdat)
		{
			pdat->fmod = onnx_attribute_read_int(n, "fmod", 0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Mod_exit(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Mod_reshape(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, a->type);
}

template <typename T>
static void Mod_int(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	T * py = (T *)y->datas;
	T * pa;
	T * pb;
	T t;

	if(pdat->fmod)
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = fmodf(*pa, *pb);
		}
	}
	else
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
			t = *pa % *pb;
			if(((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0)))
				t += *pb;
			py[i] = t;
		}
	}
}

static void Mod_int64(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	int64_t * py = (int64_t *)y->datas;
	int64_t * pa;
	int64_t * pb;
	int64_t t;

	if(pdat->fmod)
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (int64_t*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (int64_t*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = fmod(*pa, *pb);
		}
	}
	else
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pb = (int64_t*)onnx_tensor_broadcast_map_address(b, y, i);
			pa = (int64_t*)onnx_tensor_broadcast_map_address(a, y, i);
			t = *pa % *pb;
			if(((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0)))
				t += *pb;
			py[i] = t;
		}
	}
}

template <typename T>
static void Mod_uint(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	T * py = (T *)y->datas;
	T * pa;
	T * pb;

	if(pdat->fmod)
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = fmodf(*pa, *pb);
		}
	}
	else
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (T*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (T*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = *pa % *pb;
		}
	}
}

static void Mod_uint64(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * pa;
	uint64_t * pb;

	if(pdat->fmod)
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (uint64_t*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (uint64_t*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = fmod(*pa, *pb);
		}
	}
	else
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			pa = (uint64_t*)onnx_tensor_broadcast_map_address(a, y, i);
			pb = (uint64_t*)onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = *pa % *pb;
		}
	}
}

static void Mod_bfloat16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = float32_to_bfloat16(fmodf(bfloat16_to_float32(*pa), bfloat16_to_float32(*pb)));
	}
}

static void Mod_float16(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (uint16_t*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (uint16_t*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = float32_to_float16(fmodf(float16_to_float32(*pa), float16_to_float32(*pb)));
	}
}

static void Mod_float32(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	float * py = (float *)y->datas;
	float * pa;
	float * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (float*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (float*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = fmodf(*pa, *pb);
	}
}

static void Mod_float64(onnx_node_t * n)
{
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * a = n->inputs[0];
	onnx_tensor_t * b = n->inputs[1];
	double * py = (double *)y->datas;
	double * pa;
	double * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = (double*)onnx_tensor_broadcast_map_address(a, y, i);
		pb = (double*)onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = fmod(*pa, *pb);
	}
}

void resolver_default_op_Mod(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 10)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mod_init;
			n->exit = Mod_exit;
			n->reshape = Mod_reshape;
			n->ope = Mod_float64;
			break;
		default:
			break;
		}
	}
}
