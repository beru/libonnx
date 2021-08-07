#include <onnx.h>

struct ope_pdata_t {
	int axis;
	int caxis;
};

static int Concat_init(onnx_node_t * n)
{

	if((n->inputs.size() >= 1) && (n->outputs.size() == 1))
	{
		ope_pdata_t * pdat = new ope_pdata_t;
		pdat->axis = n->attribute_read_int("axis", 1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Concat_exit(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	delete pdat;
	return 1;
}

static int Concat_reshape(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x = n->inputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int * pdims;
	int i, j, s;

	pdat->caxis = pdat->axis;
	if(pdat->caxis < 0)
		pdat->caxis += ndim;
	if(pdat->caxis < 0 || pdat->caxis >= ndim)
		return 0;
	s = x->dims[pdat->caxis];
	for(i = 1; i < n->inputs.size(); i++)
	{
		pdims = n->inputs[i]->dims;
		for(j = 0; j < ndim; j++)
		{
			if(j == pdat->caxis)
				s += pdims[j];
			else if(x->dims[j] != pdims[j])
				return 0;
			dims[j] = pdims[j];
		}
	}
	dims[pdat->caxis] = s;
	return y->reshape(&dims[0], ndim, x->type);
}

static void Concat_ope(onnx_node_t * n)
{
	ope_pdata_t * pdat = (ope_pdata_t *)n->priv;
	onnx_tensor_t * y = n->outputs[0];
	onnx_tensor_t * x;
	int ybase;
	int ypitch;
	int xpitch;
	int i, j, k;
	int idx;
	size_t o, l;

	if(n->inputs[0]->type == ONNX_TENSOR_TYPE_STRING)
	{
		char ** py = (char **)y->datas;
		char ** px;
		for(i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
			ypitch *= y->dims[i];
		for(idx = 0, ybase = 0; idx < n->inputs.size(); idx++)
		{
			x = n->inputs[idx];
			px = (char **)x->datas;
			for(i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
				xpitch *= x->dims[i];
			for(o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++)
			{
				if(py[k + o])
					free(py[k + o]);
				py[k + o] = strdup(px[o]);
				if(++j == xpitch)
				{
					k += (ypitch - xpitch);
					j = 0;
				}
			}
			ybase += xpitch;
		}
	}
	else
	{
		char * py = (char *)y->datas;
		char * px;
		int sz = onnx_tensor_type_sizeof(n->inputs[0]->type);
		for(i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
			ypitch *= y->dims[i];
		for(idx = 0, ybase = 0; idx < n->inputs.size(); idx++)
		{
			x = n->inputs[idx];
			px = (char *)x->datas;
			for(i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
				xpitch *= x->dims[i];
			for(o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++)
			{
				memcpy(py + (k + o) * sz, px + o * sz, sz);
				if(++j == xpitch)
				{
					k += (ypitch - xpitch);
					j = 0;
				}
			}
			ybase += xpitch;
		}
	}
}

void resolver_default_op_Concat(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_BFLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Concat_init;
			n->exit = Concat_exit;
			n->reshape = Concat_reshape;
			n->ope = Concat_ope;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 11)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Concat_init;
			n->exit = Concat_exit;
			n->reshape = Concat_reshape;
			n->ope = Concat_ope;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 4)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_BOOL:
		case ONNX_TENSOR_TYPE_INT8:
		case ONNX_TENSOR_TYPE_INT16:
		case ONNX_TENSOR_TYPE_INT32:
		case ONNX_TENSOR_TYPE_INT64:
		case ONNX_TENSOR_TYPE_UINT8:
		case ONNX_TENSOR_TYPE_UINT16:
		case ONNX_TENSOR_TYPE_UINT32:
		case ONNX_TENSOR_TYPE_UINT64:
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
		case ONNX_TENSOR_TYPE_COMPLEX64:
		case ONNX_TENSOR_TYPE_COMPLEX128:
		case ONNX_TENSOR_TYPE_STRING:
			n->init = Concat_init;
			n->exit = Concat_exit;
			n->reshape = Concat_reshape;
			n->ope = Concat_ope;
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
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Concat_init;
			n->exit = Concat_exit;
			n->reshape = Concat_reshape;
			n->ope = Concat_ope;
			break;
		default:
			break;
		}
	}
}
