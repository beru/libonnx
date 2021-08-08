#include <onnx.h>

struct operator_pdata_t {
	int* axes;
	int naxes;
	int keepdims;

	int* caxes;
};

static int ReduceLogSumExp_init(onnx_node_t* n)
{
	int64_t* ints;
	int i;

	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		int nint = n->attribute_read_ints("axes", &ints);
		if (nint > 0)
			pdat->naxes = nint;
		else
			pdat->naxes = n->inputs[0]->ndim;
		pdat->axes = (int*)malloc(sizeof(int) * pdat->naxes);
		pdat->caxes = (int*)malloc(sizeof(int) * pdat->naxes);
		if (pdat->axes && pdat->caxes) {
			if (nint > 0) {
				for (i = 0; i < pdat->naxes; i++)
					pdat->axes[i] = ints[i];
			}else {
				for (i = 0; i < pdat->naxes; i++)
					pdat->axes[i] = i;
			}
			pdat->keepdims = n->attribute_read_int("keepdims", 1);
			n->priv = pdat;
			return 1;
		}else {
			if (pdat->axes)
				free(pdat->axes);
			if (pdat->caxes)
				free(pdat->caxes);
			delete pdat;
		}
	}
	return 0;
}

static int ReduceLogSumExp_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;

	if (pdat) {
		if (pdat->axes)
			free(pdat->axes);
		if (pdat->caxes)
			free(pdat->caxes);
		delete pdat;
	}
	return 1;
}

static int ReduceLogSumExp_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int axis, found;
	int i, j;

	for (i = 0; i < pdat->naxes; i++) {
		axis = pdat->axes[i];
		if (axis < 0)
			axis += x->ndim;
		if (axis < 0 || axis >= x->ndim)
			return 0;
		pdat->caxes[i] = axis;
	}
	if (pdat->keepdims) {
		memcpy(&dims[0], x->dims, sizeof(int) * ndim);
		for (i = 0; i < pdat->naxes; i++)
			dims[pdat->caxes[i]] = 1;
	}else {
		for (i = 0, ndim = 0; i < x->ndim; i++) {
			for (j = 0, found = 0; j < pdat->naxes; j++) {
				if (i == pdat->caxes[j]) {
					found = 1;
					break;
				}
			}
			if (!found)
				dims[ndim++]= x->dims[i];
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T, typename SumT>
static void ReduceLogSumExp_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	SumT sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	std::vector<int> iter_not_in_axes_max(not_in_axes_num);
	std::vector<int> iter_not_in_axes(not_in_axes_num);
	std::vector<int> not_in_axes_axis_dis(x->ndim);
	std::vector<int> iter_in_axes_max(pdat->naxes);
	std::vector<int> in_axes_axis_dis(pdat->naxes);
	std::vector<int> iter_in_axes(pdat->naxes);
	uint32_t mask;
	int i, j, k, o;

	for (i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for (i = 0, j = 0, k = 0; i < x->ndim; i++) {
		if (mask & (1 << i)) {
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(&iter_not_in_axes[0], 0, sizeof(int) * not_in_axes_num);
	do {
		memset(&iter_in_axes[0], 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
		sum = 0;
		do {
			sum += expf(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]);
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = logf(sum);
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

static void ReduceLogSumExp_bfloat16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	std::vector<int> iter_not_in_axes_max(not_in_axes_num);
	std::vector<int> iter_not_in_axes(not_in_axes_num);
	std::vector<int> not_in_axes_axis_dis(x->ndim);
	std::vector<int> iter_in_axes_max(pdat->naxes);
	std::vector<int> in_axes_axis_dis(pdat->naxes);
	std::vector<int> iter_in_axes(pdat->naxes);
	uint32_t mask;
	int i, j, k, o;

	for (i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for (i = 0, j = 0, k = 0; i < x->ndim; i++) {
		if (mask & (1 << i)) {
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(&iter_not_in_axes[0], 0, sizeof(int) * not_in_axes_num);
	do {
		memset(&iter_in_axes[0], 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
		sum = 0;
		do {
			sum += expf(bfloat16_to_float32(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]));
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = float32_to_bfloat16(logf(sum));
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

static void ReduceLogSumExp_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float sum;
	int not_in_axes_num = x->ndim - pdat->naxes;
	std::vector<int> iter_not_in_axes_max(not_in_axes_num);
	std::vector<int> iter_not_in_axes(not_in_axes_num);
	std::vector<int> not_in_axes_axis_dis(x->ndim);
	std::vector<int> iter_in_axes_max(pdat->naxes);
	std::vector<int> in_axes_axis_dis(pdat->naxes);
	std::vector<int> iter_in_axes(pdat->naxes);
	uint32_t mask;
	int i, j, k, o;

	for (i = 0, mask = 0; i < pdat->naxes; i++)
		mask |= (1 << pdat->caxes[i]);
	for (i = 0, j = 0, k = 0; i < x->ndim; i++) {
		if (mask & (1 << i)) {
			in_axes_axis_dis[j] = x->strides[i];
			iter_in_axes_max[j] = x->dims[i];
			j += 1;
			continue;
		}
		not_in_axes_axis_dis[k] = x->strides[i];
		iter_not_in_axes_max[k] = x->dims[i];
		k += 1;
	}
	i = 0;
	memset(&iter_not_in_axes[0], 0, sizeof(int) * not_in_axes_num);
	do {
		memset(&iter_in_axes[0], 0, sizeof(int) * pdat->naxes);
		o = dim_offset(not_in_axes_num, &iter_not_in_axes[0], &not_in_axes_axis_dis[0]);
		sum = 0;
		do {
			sum += expf(float16_to_float32(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]));
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = float32_to_float16(logf(sum));
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

void resolver_default_op_ReduceLogSumExp(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<double, double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<double, double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<int64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint8_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint32_t, float>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<uint64_t, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = ReduceLogSumExp_init;
			n->exit = ReduceLogSumExp_exit;
			n->reshape = ReduceLogSumExp_reshape;
			n->ope = ReduceLogSumExp_generic<double, double>;
			break;
		default:
			break;
		}
	}
}
