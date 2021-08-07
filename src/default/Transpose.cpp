#include <onnx.h>

struct ope_pdata_t {
	std::vector<int> perm;
};

static int Transpose_init(onnx_node_t * n)
{
	if((n->inputs.size() == 1) && (n->outputs.size() == 1))
	{
		int64_t * ints;
		ope_pdata_t * pdat = new ope_pdata_t;
		pdat->perm.resize(n->inputs[0]->ndim);
		if(pdat->perm.size() == n->attribute_read_ints("perm", &ints))
		{
			for(int i = 0; i < pdat->perm.size(); i++)
				pdat->perm[i] = ints[i];
		}
		else
		{
			for(int i = 0; i < pdat->perm.size(); i++)
				pdat->perm[i] = pdat->perm.size() - i - 1;
		}
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Transpose_exit(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	delete pdat;
	return 1;
}

static int Transpose_reshape(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int i;

	if(y->reshape_identity(x, x->type))
	{
		for(i = 0; i < x->ndim; i++)
			y->dims[i] = x->dims[pdat->perm[i]];
		return 1;
	}
	return 0;
}

template <typename T>
static void Transpose_generic(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	T * px = (T *)x->datas;
	T * py = (T *)y->datas;
	int nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		y->offset_to_indices(oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		py[oy] = px[ox];
	}
}

static void Transpose_complex64(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		y->offset_to_indices(oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_complex128(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		y->offset_to_indices(oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		py[oy] = px[ox];
		py[oy + 1] = px[ox + 1];
	}
}

static void Transpose_string(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	char ** px = (char **)x->datas;
	char ** py = (char **)y->datas;
	int nperm = pdat->perm.size();
	std::vector<int> ix(nperm), iy(nperm);
	int ox, oy;
	size_t i, l;

	for(oy = 0, l = y->ndata; oy < l; oy++)
	{
		y->offset_to_indices(oy, &iy[0]);
		for(i = 0; i < nperm; i++)
			ix[pdat->perm[i]] = iy[i];
		ox = x->indices_to_offset(&ix[0]);
		if(py[oy])
			free(py[oy]);
		py[oy] = strdup(px[ox]);
	}
}

void resolver_default_op_Transpose(onnx_node_t * n)
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
