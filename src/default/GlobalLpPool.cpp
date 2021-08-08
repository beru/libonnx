#include <onnx.h>

struct ope_pdata_t {
	float p;
};

static int GlobalLpPool_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		if (n->opset >= 2)
			pdat->p = n->attribute_read_int("p", 2);
		else
			pdat->p = n->attribute_read_float("p", 2.0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int GlobalLpPool_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int GlobalLpPool_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	for (i = 0; i < ndim; i++) {
		if (i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return y->reshape(&dims[0], ndim, x->type);
}

static void GlobalLpPool_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float v;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < C; ++j) {
			o = i * C + j;
			for (k = 0, v = float16_to_float32(0); k < m; ++k)
				v += powf(fabsf(float16_to_float32(px[o * m + k])), pdat->p);
			py[o] = float32_to_float16(powf(v, 1.0 / pdat->p));
		}
	}
}

static void GlobalLpPool_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	float* py = (float*)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < C; ++j) {
			o = i * C + j;
			for (k = 0, py[o] = 0; k < m; ++k)
				py[o] += powf(fabsf(px[o * m + k]), pdat->p);
			py[o] = powf(py[o], 1.0 / pdat->p);
		}
	}
}

static void GlobalLpPool_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	double* py = (double*)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < C; ++j) {
			o = i * C + j;
			for (k = 0, py[o] = 0; k < m; ++k)
				py[o] += pow(fabs(px[o * m + k]), pdat->p);
			py[o] = pow(py[o], 1.0 / pdat->p);
		}
	}
}

void resolver_default_op_GlobalLpPool(onnx_node_t* n)
{
	if (n->opset >= 2) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->ope = GlobalLpPool_float64;
			break;
		default:
			break;
		}
	}
}
