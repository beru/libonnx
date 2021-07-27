#include <onnx.h>

struct ope_pdata_t {
	int * perm;
	int nperm;
};

static int Transpose_init(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat;
	int64_t * ints;
	int i;

	if((n->ninput == 1) && (n->noutput == 1))
	{
		pdat = (struct ope_pdata_t *)malloc(sizeof(struct ope_pdata_t));
		if(pdat)
		{
			pdat->nperm = n->inputs[0]->ndim;
			pdat->perm = (int*)malloc(sizeof(int) * pdat->nperm);
			if(pdat->perm)
			{
				if(pdat->nperm == onnx_attribute_read_ints(n, "perm", &ints))
				{
					for(i = 0; i < pdat->nperm; i++)
						pdat->perm[i] = ints[i];
				}
				else
				{
					for(i = 0; i < pdat->nperm; i++)
						pdat->perm[i] = pdat->nperm - i - 1;
				}
				n->priv = pdat;
				return 1;
			}
			else
			{
				free(pdat);
			}
		}
	}
	return 0;
}

static int Transpose_exit(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->perm)
			free(pdat->perm);
		free(pdat);
	}
	return 1;
}

static int Transpose_reshape(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	int i;

	if(onnx_tensor_reshape_identity(y, x, x->type))
	{
		for(i = 0; i < x->ndim; i++)
			y->dims[i] = x->dims[pdat->perm[i]];
		return 1;
	}
	return 0;
}

template <typename T>
static void Transpose_generic(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	T * px = (T *)x->datas;
	T * py = (T *)y->datas;
	int nperm = pdat->nperm;
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, &ix[0]);
		py[oy] = px[ox];
	}
}

static void Transpose_complex64(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int nperm = pdat->nperm;
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, &ix[0]);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_complex128(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int nperm = pdat->nperm;
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, &ix[0]);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_string(struct onnx_node_t * n)
{
	struct ope_pdata_t * pdat = (struct ope_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;
	int nperm = pdat->nperm;
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		onnx_tensor_offset_to_indices(y, oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = onnx_tensor_indices_to_offset(x, &ix[0]);
		if(py[oy])
			free(py[oy]);
		py[oy] = strdup(px[ox]);
	}
}

void resolver_default_op_Transpose(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<double>;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_string;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int16_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int32_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint32_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_generic<double>;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_complex64;
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_complex128;
			break;
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Transpose_init;
			n->exit = Transpose_exit;
			n->reshape = Transpose_reshape;
			n->ope = Transpose_string;
			break;
		default:
			break;
		}
	}
}
