#include <onnx.h>

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

struct operator_pdata_t {
	auto_pad_t auto_pad;
	int ceil_mode;
	int storage_order;
	int* kernels;
	int nkernel;
	int* dilations;
	int ndilation;
	int* pads;
	int npad;
	int* strides;
	int nstride;

	int cpads[32];
};

static int MaxPool_init(onnx_node_t* n)
{
	int64_t* ints;
	int i, l;

	if ((n->inputs.size() == 1) && (n->outputs.size() >= 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		memset(pdat, 0, sizeof(operator_pdata_t));
		switch (C_HASH(n->attribute_read_string("auto_pad", "NOTSET")))	{
		case C_HASH("NOTSET"):
			pdat->auto_pad = AUTO_PAD_NOTSET;
			break;
		case C_HASH("SAME_UPPER"):
			pdat->auto_pad = AUTO_PAD_SAME_UPPER;
			break;
		case C_HASH("SAME_LOWER"):
			pdat->auto_pad = AUTO_PAD_SAME_LOWER;
			break;
		case C_HASH("VALID"):
			pdat->auto_pad = AUTO_PAD_VALID;
			break;
		default:
			pdat->auto_pad = AUTO_PAD_NOTSET;
			break;
		}
		pdat->ceil_mode = n->attribute_read_int("ceil_mode", 0);
		pdat->storage_order = n->attribute_read_int("storage_order", 0);
		pdat->nkernel = n->attribute_read_ints("kernel_shape", &ints);
		if (pdat->nkernel > 0) {
			pdat->kernels = (int*)malloc(sizeof(int) * pdat->nkernel);
			for (i = 0; i < pdat->nkernel; i++)
				pdat->kernels[i] = ints[i];
		}
		pdat->ndilation = pdat->nkernel;
		pdat->dilations = (int*)malloc(sizeof(int) * pdat->ndilation);
		if (pdat->dilations) {
			l = n->attribute_read_ints("dilations", &ints);
			for (i = 0; i < l; i++)
				pdat->dilations[i] = ints[i];
			for (; i < pdat->ndilation; i++)
				pdat->dilations[i] = 1;
		}
		pdat->npad = pdat->nkernel * 2;
		pdat->pads = (int*)malloc(sizeof(int) * pdat->npad);
		if (pdat->pads) {
			l = n->attribute_read_ints("pads", &ints);
			for (i = 0; i < l; i++)
				pdat->pads[i] = ints[i];
			for (; i < pdat->npad; i++)
				pdat->pads[i] = 0;
		}
		pdat->nstride = pdat->nkernel;
		pdat->strides = (int*)malloc(sizeof(int) * pdat->nstride);
		if (pdat->strides) {
			l = n->attribute_read_ints("strides", &ints);
			for (i = 0; i < l; i++)
				pdat->strides[i] = ints[i];
			for (; i < pdat->nstride; i++)
				pdat->strides[i] = 1;
		}
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int MaxPool_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;

	if (pdat) {
		if (pdat->kernels)
			free(pdat->kernels);
		if (pdat->dilations)
			free(pdat->dilations);
		if (pdat->pads)
			free(pdat->pads);
		if (pdat->strides)
			free(pdat->strides);
		delete pdat;
	}
	return 1;
}

static int MaxPool_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int pad;
	int i;

	switch (pdat->auto_pad) {
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, pdat->pads, sizeof(int) * pdat->npad);
		break;
	case AUTO_PAD_SAME_UPPER:
		for (i = 0; i < pdat->npad / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i] = pad / 2;
			pdat->cpads[i + pdat->nkernel] = pad - pdat->cpads[i];
		}
		break;
	case AUTO_PAD_SAME_LOWER:
		for (i = 0; i < pdat->npad / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i + pdat->nkernel] = pad / 2;
			pdat->cpads[i] = pad - pdat->cpads[i + pdat->nkernel];
		}
		break;
	case AUTO_PAD_VALID:
		memset(pdat->cpads, 0, sizeof(int) * pdat->npad);
		break;
	default:
		break;
	}
	dims[0] = x->dims[0];
	dims[1] = x->dims[1];
	for (i = 0; i < ndim - 2; i++) {
		switch (pdat->auto_pad)	{
		case AUTO_PAD_NOTSET:
			if (pdat->ceil_mode)
				dims[i + 2] = ceilf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			else
				dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->nkernel] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			break;
		case AUTO_PAD_SAME_UPPER:
		case AUTO_PAD_SAME_LOWER:
			dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
			break;
		case AUTO_PAD_VALID:
			dims[i + 2] = ceilf((x->dims[i + 2] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) + 1) / (float)pdat->strides[i]);
			break;
		default:
			break;
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
static void MaxPool_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	T maxv, v;
	std::vector<int> k_dim(x->ndim - 2);
	std::vector<int> i_dim(x->ndim);
	std::vector<int> o_dim(x->ndim);
	std::vector<int> b_dim(x->ndim);
	int i;

	memset(&o_dim[0], 0, sizeof(o_dim));
	do {
		for (i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = std::numeric_limits<T>::min();
		memset(&k_dim[0], 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for (i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for (i = 0; i < x->ndim; ++i) {
				if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if (i >= x->ndim) {
				v = px[dim_offset(x->ndim, &i_dim[0], x->dims)];
				maxv = max(v, maxv);
			}
		} while (dim_next(x->ndim - 2, &k_dim[0], pdat->kernels));
		py[dim_offset(x->ndim, &o_dim[0], y->dims)] = maxv;
	} while (dim_next(x->ndim, &o_dim[0], y->dims));
}

static void MaxPool_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float maxv, v;
	std::vector<int> k_dim(x->ndim - 2);
	std::vector<int> i_dim(x->ndim);
	std::vector<int> o_dim(x->ndim);
	std::vector<int> b_dim(x->ndim);
	int i;

	memset(&o_dim[0], 0, sizeof(o_dim));
	do {
		for (i = 2; i < x->ndim; ++i)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		maxv = -FLT_MAX;
		memset(&k_dim[0], 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for (i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			for (i = 0; i < x->ndim; ++i) {
				if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
					break;
			}
			if (i >= x->ndim) {
				v = float16_to_float32(px[dim_offset(x->ndim, &i_dim[0], x->dims)]);
				maxv = fmaxf(v, maxv);
			}
		} while (dim_next(x->ndim - 2, &k_dim[0], pdat->kernels));
		py[dim_offset(x->ndim, &o_dim[0], y->dims)] = float32_to_float16(maxv);
	} while (dim_next(x->ndim, &o_dim[0], y->dims));
}

void resolver_default_op_MaxPool(onnx_node_t* n)
{
	if (n->opset >= 12) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<int8_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<uint8_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 10) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 8) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = MaxPool_init;
			n->exit = MaxPool_exit;
			n->reshape = MaxPool_reshape;
			n->ope = MaxPool_generic<double>;
			break;
		default:
			break;
		}
	}
}
