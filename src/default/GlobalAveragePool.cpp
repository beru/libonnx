#include <onnx.h>
#include "refnd.h"

static int GlobalAveragePool_init(onnx_node_t * n)
{
	if((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int GlobalAveragePool_exit(onnx_node_t * n)
{
	return 1;
}

static int GlobalAveragePool_reshape(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	for(i = 0; i < ndim; i++)
	{
		if(i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
static void GlobalAveragePool_generic(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	T * px = (T *)x->datas;
	T * py = (T *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int avgsz = x->ndata / (N * C);
	std::vector<T> buf(N * C);
	ref2d<T> sum(C, &buf[0]);
	int idx[2], cnt;
	size_t i, j, l;

	for(i = 0, l = x->ndata; i < l; i++)
	{
		cnt = i;
		idx[0] = cnt / x->strides[0];
		cnt %= x->strides[0];
		idx[1] = cnt / x->strides[1];
		cnt %= x->strides[1];
		sum[idx[0]][idx[1]] += px[i];
	}
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < C; j++)
			py[i * C + j] = sum[i][j] / avgsz;
	}
}

void resolver_default_op_GlobalAveragePool(onnx_node_t * n)
{
	if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = GlobalAveragePool_init;
			n->exit = GlobalAveragePool_exit;
			n->reshape = GlobalAveragePool_reshape;
			n->ope = GlobalAveragePool_generic<uint16_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = GlobalAveragePool_init;
			n->exit = GlobalAveragePool_exit;
			n->reshape = GlobalAveragePool_reshape;
			n->ope = GlobalAveragePool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = GlobalAveragePool_init;
			n->exit = GlobalAveragePool_exit;
			n->reshape = GlobalAveragePool_reshape;
			n->ope = GlobalAveragePool_generic<double>;
			break;
		default:
			break;
		}
	}
}
