#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

struct operator_pdata_t : public node_t::ope_pdata_t {
	~operator_pdata_t() {
	}

	auto_pad_t auto_pad;
	int ceil_mode = 0;
	int storage_order = 0;
	std::vector<int> kernels;
	int nkernel = 0;
	std::vector<int> dilations;
	int ndilation = 0;
	std::vector<int> pads;
	int npad = 0;
	std::vector<int> strides;
	int nstride = 0;

	int cpads[32] = {0};
};

bool MaxPool_init(node_t* n)
{
	if (!(n->inputs.size() == 1 && n->outputs.size() >= 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	if (!pdat)
		return false;
	int64_t* ints;
	int i, l;
	switch (C_HASH(n->attribute("auto_pad", "NOTSET")))	{
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
	pdat->ceil_mode = n->attribute("ceil_mode", 0);
	pdat->storage_order = n->attribute("storage_order", 0);
	pdat->nkernel = n->attribute("kernel_shape", &ints);
	if (pdat->nkernel > 0) {
		pdat->kernels.resize(pdat->nkernel);
		for (i = 0; i < pdat->nkernel; i++)
			pdat->kernels[i] = ints[i];
	}
	pdat->ndilation = pdat->nkernel;
	pdat->dilations.resize(pdat->ndilation);
	if (pdat->ndilation > 0) {
		l = n->attribute("dilations", &ints);
		for (i = 0; i < l; i++)
			pdat->dilations[i] = ints[i];
		for (; i < pdat->ndilation; i++)
			pdat->dilations[i] = 1;
	}
	pdat->npad = pdat->nkernel * 2;
	pdat->pads.resize(pdat->npad);
	if (pdat->npad > 0) {
		l = n->attribute("pads", &ints);
		for (i = 0; i < l; i++)
			pdat->pads[i] = ints[i];
		for (; i < pdat->npad; i++)
			pdat->pads[i] = 0;
	}
	pdat->nstride = pdat->nkernel;
	pdat->strides.resize(pdat->nstride);
	if (pdat->nstride > 0) {
		l = n->attribute("strides", &ints);
		for (i = 0; i < l; i++)
			pdat->strides[i] = ints[i];
		for (; i < pdat->nstride; i++)
			pdat->strides[i] = 1;
	}
	n->priv = pdat;
	return true;
}

int MaxPool_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int pad;
	int i;

	switch (pdat->auto_pad) {
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, &pdat->pads[0], sizeof(int) * pdat->npad);
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
void MaxPool_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
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
				v = px[dim_offset(x->ndim, &i_dim[0], &x->dims[0])];
				maxv = max(v, maxv);
			}
		} while (dim_next(x->ndim - 2, &k_dim[0], &pdat->kernels[0]));
		py[dim_offset(x->ndim, &o_dim[0], &y->dims[0])] = maxv;
	} while (dim_next(x->ndim, &o_dim[0], &y->dims[0]));
}

GEN_HOLEDR_TYPE(holder, MaxPool_generic)

} // namespace

void resolver_default_op_MaxPool(node_t* n)
{
	if (n->opset >= 12) {
		n->ope = ope_type_select<holder,
			int8_t,
			uint8_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = MaxPool_init;
		n->reshape = MaxPool_reshape;
	}
}

} // namespace onnx
