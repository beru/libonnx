#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_13_pdata_t : public node_t::ope_pdata_t {
	int axis;

	int caxis;
	int current;
	int outter;
	int inner;
};

bool LogSoftmax_13_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_13_pdata_t>();
	pdat->axis = n->attribute("axis", -1);
	n->priv = pdat;
	return true;
}

int LogSoftmax_13_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_13_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int i;

	pdat->caxis = pdat->axis;
	if (pdat->caxis < 0)
		pdat->caxis += x->ndim;
	if (pdat->caxis < 0 || pdat->caxis >= x->ndim)
		return 0;
	for (i = 0, pdat->outter = 1, pdat->inner = 1; i < x->ndim; i++) {
		if (i == pdat->caxis)
			pdat->current = x->dims[i];
		else if (i < pdat->caxis)
			pdat->outter *= x->dims[i];
		else
			pdat->inner *= x->dims[i];
	}
	return y->reshape_identity(x);
}

template <typename T>
void LogSoftmax_13_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_13_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
	T* py = (T*)y->data;
	T maxv, sum;
	int i, j, k, o, oo, io;

	for (i = 0; i < pdat->outter; i++) {
		oo = i * pdat->current * pdat->inner;
		for (k = 0; k < pdat->inner; k++) {
			io = oo + k;
			for (j = 0, maxv = px[io]; j < pdat->current; j++) {
				o = io + j * pdat->inner;
				if (px[o] > maxv)
					maxv = px[o];
			}
			for (j = 0, sum = 0; j < pdat->current; j++) {
				o = io + j * pdat->inner;
				py[o] = exp(px[o] - maxv);
				sum += py[o];
			}
			if (sum != 0) {
				for (j = 0; j < pdat->current; j++) {
					io = oo + j * pdat->inner + k;
					py[io] = log(py[io] / sum);
				}
			}
		}
	}
}

struct operator_1_11_pdata_t : public node_t::ope_pdata_t {
	int axis;

	int N;
	int D;
};

bool LogSoftmax_1_11_init(node_t* n)
{
	if (!(n->inputs.size() == 1 && n->outputs.size() == 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_1_11_pdata_t>();
	pdat->axis = n->attribute("axis", 1);
	n->priv = pdat;
	return true;
}

int LogSoftmax_1_11_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_1_11_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int axis = pdat->axis;
	int i;

	if (axis < 0)
		axis += x->ndim;
	if (axis < 0 || axis >= x->ndim)
		return 0;
	for (i = 0, pdat->N = 1, pdat->D = 1; i < x->ndim; i++) {
		if (i < axis)
			pdat->N *= x->dims[i];
		else
			pdat->D *= x->dims[i];
	}
	return y->reshape_identity(x);
}

template <typename T>
void LogSoftmax_1_11_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_1_11_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
	T* py = (T*)y->data;
	T maxv, sum;
	int i, j, o;

	for (i = 0, o = 0; i < pdat->N; i++, o += pdat->D) {
		for (j = 0, maxv = std::numeric_limits<T>::min(); j < pdat->D; j++) {
			if (px[o + j] > maxv)
				maxv = px[o + j];
		}
		for (j = 0, sum = 0; j < pdat->D; j++) {
			py[o + j] = exp(px[o + j] - maxv);
			sum += py[o + j];
		}
		if (sum != 0) {
			for (j = 0; j < pdat->D; j++)
				py[o + j] = log(py[o + j] / sum);
		}
	}
}

GEN_HOLEDR_TYPE(holder_13, LogSoftmax_13_generic)
GEN_HOLEDR_TYPE(holder_1_11, LogSoftmax_1_11_generic)

} // namespace

void resolver_default_op_LogSoftmax(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder_13,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_13_init;
			n->reshape = LogSoftmax_13_reshape;
		}
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder_1_11,
			float16_t, float, double
		>(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_1_11_init;
			n->reshape = LogSoftmax_1_11_reshape;
		}
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder_1_11,
			float16_t, float, double
		>(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_1_11_init;
			n->reshape = LogSoftmax_1_11_reshape;
		}
	}
}

} // namespace onnx
