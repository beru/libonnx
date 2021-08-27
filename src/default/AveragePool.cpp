#include <onnx.h>
#include "util.h"

namespace {

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

struct ope_pdata_t : public onnx_node_t::ope_pdata_t {
	auto_pad_t auto_pad;
	int ceil_mode;
	int count_include_pad;
	std::vector<int> kernels;
	std::vector<int> pads;
	std::vector<int> strides;

	int cpads[32];
};

bool AveragePool_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	int i, l;
	int64_t* ints;
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	memset(pdat, 0, sizeof(ope_pdata_t));
	switch (C_HASH(n->attribute_read_string("auto_pad", "NOTSET"))) {
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
	pdat->count_include_pad = n->attribute_read_int("count_include_pad", 0);
	pdat->kernels.resize(n->attribute_read_ints("kernel_shape", &ints));
	for (i = 0; i < pdat->kernels.size(); i++)
		pdat->kernels[i] = ints[i];
	pdat->pads.resize(pdat->kernels.size() * 2);
	if (pdat->pads.size()) {
		l = n->attribute_read_ints("pads", &ints);
		for (i = 0; i < l; i++)
			pdat->pads[i] = ints[i];
		for (; i < pdat->pads.size(); i++)
			pdat->pads[i] = 0;
	}
	pdat->strides.resize(pdat->kernels.size());
	if (pdat->strides.size()) {
		l = n->attribute_read_ints("strides", &ints);
		for (i = 0; i < l; i++)
			pdat->strides[i] = ints[i];
		for (; i < pdat->strides.size(); i++)
			pdat->strides[i] = 1;
	}
	n->priv = pdat;
	return true;
}

int AveragePool_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int pad;
	int i;

	switch (pdat->auto_pad) {
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, &pdat->pads[0], sizeof(int) * pdat->pads.size());
		break;
	case AUTO_PAD_SAME_UPPER:
		for (i = 0; i < pdat->pads.size() / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + pdat->kernels[i] - x->dims[i + 2];
			pdat->cpads[i] = pad / 2;
			pdat->cpads[i + pdat->kernels.size()] = pad - pdat->cpads[i];
		}
		break;
	case AUTO_PAD_SAME_LOWER:
		for (i = 0; i < pdat->pads.size() / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + pdat->kernels[i] - x->dims[i + 2];
			pdat->cpads[i + pdat->kernels.size()] = pad / 2;
			pdat->cpads[i] = pad - pdat->cpads[i + pdat->kernels.size()];
		}
		break;
	case AUTO_PAD_VALID:
		memset(pdat->cpads, 0, sizeof(int) * pdat->pads.size());
		break;
	default:
		break;
	}
	dims[0] = x->dims[0];
	dims[1] = x->dims[1];
	for (i = 0; i < ndim - 2; i++) {
		switch (pdat->auto_pad) {
		case AUTO_PAD_NOTSET:
			if (pdat->ceil_mode)
				dims[i + 2] = ceilf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->kernels.size()] - pdat->kernels[i]) / (float)pdat->strides[i] + 1);
			else
				dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->kernels.size()] - pdat->kernels[i]) / (float)pdat->strides[i] + 1);
			break;
		case AUTO_PAD_SAME_UPPER:
		case AUTO_PAD_SAME_LOWER:
			dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
			break;
		case AUTO_PAD_VALID:
			dims[i + 2] = ceilf((x->dims[i + 2] - pdat->kernels[i] + 1) / (float)pdat->strides[i]);
			break;
		default:
			break;
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
void AveragePool_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
	T sum;
	std::vector<int> k_dim(x->ndim - 2);
	std::vector<int> i_dim(x->ndim);
	std::vector<int> o_dim(x->ndim);
	std::vector<int> b_dim(x->ndim);
	int padcnt, ispad, size;
	int i;

	for (i = 0, size = 1; i < x->ndim - 2; ++i) {
		size *= pdat->kernels[i];
	}
	memset(&o_dim[0], 0, sizeof(o_dim));
	do {
		for (i = 2; i < x->ndim; i++)
			b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
		sum = 0;
		padcnt = 0;
		memset(&k_dim[0], 0, sizeof(k_dim));
		do {
			i_dim[0] = o_dim[0];
			i_dim[1] = o_dim[1];
			for (i = 2; i < x->ndim; ++i)
				i_dim[i] = b_dim[i] + k_dim[i - 2];
			ispad = 0;
			for (i = 0; i < x->ndim; i++) {
				if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
					ispad = 1;
					break;
				}
			}
			if (i >= x->ndim)
				sum += px[dim_offset(x->ndim, &i_dim[0], x->dims)];
			if (ispad)
				padcnt++;
		} while (dim_next(x->ndim - 2, &k_dim[0], &pdat->kernels[0]));
		if (pdat->count_include_pad)
			sum /= size;
		else
			sum /= (size - padcnt);
		py[dim_offset(x->ndim, &o_dim[0], y->dims)] = sum;
	} while (dim_next(x->ndim, &o_dim[0], y->dims));
}

GEN_HOLEDR_TYPE(holder, AveragePool_generic)

} // namespace

void resolver_default_op_AveragePool(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = AveragePool_init;
		n->reshape = AveragePool_reshape;
	}
}
