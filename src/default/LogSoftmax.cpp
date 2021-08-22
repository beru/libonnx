#include <onnx.h>

struct operator_13_pdata_t
{
	int axis;

	int caxis;
	int current;
	int outter;
	int inner;
};

static int LogSoftmax_13_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_13_pdata_t* pdat = new operator_13_pdata_t;
		pdat->axis = n->attribute_read_int("axis", -1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int LogSoftmax_13_exit(onnx_node_t* n)
{
	operator_13_pdata_t* pdat = (operator_13_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int LogSoftmax_13_reshape(onnx_node_t* n)
{
	operator_13_pdata_t* pdat = (operator_13_pdata_t*)n->priv;
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

static void LogSoftmax_13_bfloat16(onnx_node_t* n)
{
	operator_13_pdata_t* pdat = (operator_13_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float maxv, sum, v;
	int i, j, k, o, oo, io;

	for (i = 0; i < pdat->outter; i++) {
		oo = i * pdat->current * pdat->inner;
		for (k = 0; k < pdat->inner; k++) {
			io = oo + k;
			for (j = 0, maxv = bfloat16_to_float32(px[io]); j < pdat->current; j++) {
				o = io + j * pdat->inner;
				v = bfloat16_to_float32(px[o]);
				if (v > maxv)
					maxv = v;
			}
			for (j = 0, sum = 0; j < pdat->current; j++) {
				o = io + j * pdat->inner;
				v = expf(bfloat16_to_float32(px[o]) - maxv);
				py[o] = float32_to_bfloat16(v);
				sum += v;
			}
			if (sum != 0) {
				for (j = 0; j < pdat->current; j++) {
					io = oo + j * pdat->inner + k;
					v = bfloat16_to_float32(py[io]);
					py[io] = float32_to_bfloat16(logf(v / sum));
				}
			}
		}
	}
}

static void LogSoftmax_13_float16(onnx_node_t* n)
{
	operator_13_pdata_t* pdat = (operator_13_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float maxv, sum, v;
	int i, j, k, o, oo, io;

	for (i = 0; i < pdat->outter; i++) {
		oo = i * pdat->current * pdat->inner;
		for (k = 0; k < pdat->inner; k++) {
			io = oo + k;
			for (j = 0, maxv = float16_to_float32(px[io]); j < pdat->current; j++) {
				o = io + j * pdat->inner;
				v = float16_to_float32(px[o]);
				if (v > maxv)
					maxv = v;
			}
			for (j = 0, sum = 0; j < pdat->current; j++) {
				o = io + j * pdat->inner;
				v = expf(float16_to_float32(px[o]) - maxv);
				py[o] = float32_to_float16(v);
				sum += v;
			}
			if (sum != 0) {
				for (j = 0; j < pdat->current; j++) {
					io = oo + j * pdat->inner + k;
					v = float16_to_float32(py[io]);
					py[io] = float32_to_float16(logf(v / sum));
				}
			}
		}
	}
}

template <typename T>
static void LogSoftmax_13_generic(onnx_node_t* n)
{
	operator_13_pdata_t* pdat = (operator_13_pdata_t*)n->priv;
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
					py[io] = log(py[io] / sum);
				}
			}
		}
	}
}

struct operator_1_11_pdata_t {
	int axis;

	int N;
	int D;
};

static int LogSoftmax_1_11_init(onnx_node_t* n)
{

	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_1_11_pdata_t * pdat = new operator_1_11_pdata_t;
		pdat->axis = n->attribute_read_int("axis", 1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int LogSoftmax_1_11_exit(onnx_node_t* n)
{
	operator_1_11_pdata_t * pdat = (operator_1_11_pdata_t *)n->priv;
	delete pdat;
	return 1;
}

static int LogSoftmax_1_11_reshape(onnx_node_t* n)
{
	operator_1_11_pdata_t * pdat = (operator_1_11_pdata_t *)n->priv;
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

static void LogSoftmax_1_11_float16(onnx_node_t* n)
{
	operator_1_11_pdata_t * pdat = (operator_1_11_pdata_t *)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float maxv, sum, v;
	int i, j, o;

	for (i = 0, o = 0; i < pdat->N; i++, o += pdat->D) {
		for (j = 0, maxv = FLT_MIN; j < pdat->D; j++) {
			v = float16_to_float32(px[o + j]);
			if (v > maxv)
				maxv = v;
		}
		for (j = 0, sum = 0; j < pdat->D; j++) {
			v = expf(float16_to_float32(px[o + j]) - maxv);
			py[o + j] = float32_to_float16(v);
			sum += v;
		}
		if (sum != 0) {
			for (j = 0; j < pdat->D; j++) {
				v = float16_to_float32(py[o + j]);
				py[o + j] = float32_to_float16(logf(v / sum));
			}
		}
	}
}

template <typename T>
static void LogSoftmax_1_11_generic(onnx_node_t* n)
{
	operator_1_11_pdata_t * pdat = (operator_1_11_pdata_t *)n->priv;
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
				py[o + j] = log(py[o + j] / sum);
		}
	}
}

void resolver_default_op_LogSoftmax(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = LogSoftmax_13_bfloat16,
			.float16_ = LogSoftmax_13_float16,
			.float32_ = LogSoftmax_13_generic<float>,
			.float64_ = LogSoftmax_13_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_13_init;
			n->exit = LogSoftmax_13_exit;
			n->reshape = LogSoftmax_13_reshape;
		}
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.float16_ = LogSoftmax_1_11_float16,
			.float32_ = LogSoftmax_1_11_generic<float>,
			.float64_ = LogSoftmax_1_11_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_1_11_init;
			n->exit = LogSoftmax_1_11_exit;
			n->reshape = LogSoftmax_1_11_reshape;
		}
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = LogSoftmax_1_11_float16,
			.float32_ = LogSoftmax_1_11_generic<float>,
			.float64_ = LogSoftmax_1_11_generic<double>,
		}.select(n->inputs[0]->type);
		if (n->ope) {
			n->init = LogSoftmax_1_11_init;
			n->exit = LogSoftmax_1_11_exit;
			n->reshape = LogSoftmax_1_11_reshape;
		}
	}
}
