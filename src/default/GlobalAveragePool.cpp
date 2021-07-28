#include <onnx.h>

static int GlobalAveragePool_init(onnx_node_t * n)
{
	if((n->ninput == 1) && (n->noutput == 1))
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
	return onnx_tensor_reshape(y, &dims[0], ndim, x->type);
}

static void GlobalAveragePool_float16(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int avgsz = x->ndata / (N * C);
	float sum[N][C];
	int idx[2], cnt;
	size_t i, j, l;

	memset(sum, 0, sizeof(sum));
	for(i = 0, l = x->ndata; i < l; i++)
	{
		cnt = i;
		idx[0] = cnt / x->strides[0];
		cnt %= x->strides[0];
		idx[1] = cnt / x->strides[1];
		cnt %= x->strides[1];
		sum[idx[0]][idx[1]] += float16_to_float32(px[i]);
	}
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < C; j++)
			py[i * C + j] = float32_to_float16(sum[i][j] / avgsz);
	}
}

static void GlobalAveragePool_float32(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int avgsz = x->ndata / (N * C);
	float sum[N][C];
	int idx[2], cnt;
	size_t i, j, l;

	memset(sum, 0, sizeof(sum));
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

static void GlobalAveragePool_float64(onnx_node_t * n)
{
	onnx_tensor_t * x = n->inputs[0];
	onnx_tensor_t * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int avgsz = x->ndata / (N * C);
	double sum[N][C];
	int idx[2], cnt;
	size_t i, j, l;

	memset(sum, 0, sizeof(sum));
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
			n->ope = GlobalAveragePool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = GlobalAveragePool_init;
			n->exit = GlobalAveragePool_exit;
			n->reshape = GlobalAveragePool_reshape;
			n->ope = GlobalAveragePool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = GlobalAveragePool_init;
			n->exit = GlobalAveragePool_exit;
			n->reshape = GlobalAveragePool_reshape;
			n->ope = GlobalAveragePool_float64;
			break;
		default:
			break;
		}
	}
}
