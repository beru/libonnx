#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

struct ope_13_pdata_t
{
	int axis;

	int caxis;
	int current;
	int outter;
	int inner;
};

static int Softmax_13_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_13_pdata_t* pdat = new ope_13_pdata_t;
		pdat->axis = n->attribute_read_int("axis", -1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Softmax_13_exit(onnx_node_t* n)
{
	ope_13_pdata_t* pdat = (ope_13_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Softmax_13_reshape(onnx_node_t* n)
{
	ope_13_pdata_t* pdat = (ope_13_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
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
	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Softmax_13_generic(onnx_node_t* n)
{
	ope_13_pdata_t* pdat = (ope_13_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
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
					py[io] /= sum;
				}
			}
		}
	}
}

struct ope_1_11_pdata_t {
	int axis;

	int N;
	int D;
};

static int Softmax_1_11_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_1_11_pdata_t* pdat = new ope_1_11_pdata_t;
		pdat->axis = n->attribute_read_int("axis", 1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Softmax_1_11_exit(onnx_node_t* n)
{
	ope_1_11_pdata_t* pdat = (ope_1_11_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Softmax_1_11_reshape(onnx_node_t* n)
{
	ope_1_11_pdata_t* pdat = (ope_1_11_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
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
	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Softmax_1_11_generic(onnx_node_t* n)
{
	ope_1_11_pdata_t* pdat = (ope_1_11_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
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
				py[o + j] /= sum;
		}
	}
}

void resolver_default_op_Softmax(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Softmax_13_generic<bfloat16_t>,
			.float16_ = Softmax_13_generic<float16_t>,
			.float32_ = Softmax_13_generic<float>,
			.float64_ = Softmax_13_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = Softmax_13_init;
			n->exit = Softmax_13_exit;
			n->reshape = Softmax_13_reshape;
		}
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Softmax_1_11_generic<float16_t>,
			.float32_ = Softmax_1_11_generic<float>,
			.float64_ = Softmax_1_11_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
		}
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Softmax_1_11_generic<float16_t>,
			.float32_ = Softmax_1_11_generic<float>,
			.float64_ = Softmax_1_11_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = Softmax_1_11_init;
			n->exit = Softmax_1_11_exit;
			n->reshape = Softmax_1_11_reshape;
		}
	}
}
