#include <onnx.h>

struct operator_pdata_t {
	int keepdims;
	int noop_with_empty_axes;

	int caxes[32];
	int naxes;
};

static int ReduceSum_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->keepdims = n->attribute_read_int("keepdims", 1);
		pdat->noop_with_empty_axes = n->attribute_read_int("noop_with_empty_axes", 0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int ReduceSum_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int ReduceSum_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int axis, found;
	int i, j;

	if ((n->inputs.size() > 1) && (n->inputs[1]->ndata > 0)) {
		onnx_tensor_t* a = n->inputs[1];
		int64_t* pa = (int64_t*)a->datas;
		pdat->naxes = min(min(x->ndim, 32), (int)a->ndata);
		for (i = 0; i < pdat->naxes; i++) {
			axis = pa[i];
			if (axis < 0)
				axis += x->ndim;
			if (axis < 0 || axis >= x->ndim)
				return 0;
			pdat->caxes[i] = axis;
		}
	}else if (pdat->noop_with_empty_axes == 0) {
		pdat->naxes = min(x->ndim, 32);
		for (i = 0; i < pdat->naxes; i++)
			pdat->caxes[i] = i;
	}else {
		pdat->naxes = 0;
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
static void ReduceSum_generic(onnx_node_t* n)
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
			sum += px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])];
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = sum;
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

static void ReduceSum_bfloat16(onnx_node_t* n)
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
			sum += bfloat16_to_float32(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]);
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = float32_to_bfloat16(sum);
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

static void ReduceSum_float16(onnx_node_t* n)
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
			sum += float16_to_float32(px[o + dim_offset(pdat->naxes, &iter_in_axes[0], &in_axes_axis_dis[0])]);
		} while (dim_next(pdat->naxes, &iter_in_axes[0], &iter_in_axes_max[0]));
		py[i++] = float32_to_float16(sum);
	} while (dim_next(not_in_axes_num, &iter_not_in_axes[0], &iter_not_in_axes_max[0]));
}

void resolver_default_op_ReduceSum(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = ReduceSum_generic<int8_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = ReduceSum_generic<int32_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = ReduceSum_generic<int64_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = ReduceSum_generic<uint8_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = ReduceSum_generic<uint32_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = ReduceSum_generic<uint64_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = ReduceSum_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = ReduceSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = ReduceSum_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = ReduceSum_generic<double, double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = ReduceSum_generic<int8_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = ReduceSum_generic<int32_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = ReduceSum_generic<int64_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = ReduceSum_generic<uint8_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = ReduceSum_generic<uint32_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = ReduceSum_generic<uint64_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = ReduceSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = ReduceSum_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = ReduceSum_generic<double, double>;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_INT8:
			n->ope = ReduceSum_generic<int8_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->ope = ReduceSum_generic<int32_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->ope = ReduceSum_generic<int64_t, int64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->ope = ReduceSum_generic<uint8_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->ope = ReduceSum_generic<uint32_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->ope = ReduceSum_generic<uint64_t, uint64_t>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = ReduceSum_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = ReduceSum_generic<float, float>;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = ReduceSum_generic<double, double>;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = ReduceSum_init;
		n->exit = ReduceSum_exit;
		n->reshape = ReduceSum_reshape;
	}
}
