#include <onnx.h>

struct operator_pdata_t {
	float alpha;
	float beta;
	float bias;
	int size;
};

static int LRN_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 0.0001);
		pdat->beta = n->attribute_read_float("beta", 0.75);
		pdat->bias = n->attribute_read_float("bias", 1.0);
		pdat->size = n->attribute_read_int("size", 1);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int LRN_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int LRN_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

static void LRN_bfloat16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float sum, t;
	float over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for (u = 0; u < N; u++) {
		for (v = 0; v < C; v++) {
			for (i = 0; i < L; i++) {
				start = v - (pdat->size / 2);
				if (start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if (end >= C)
					end = C - 1;
				for (j = start, sum = 0; j <= end; ++j) {
					t = bfloat16_to_float32(px[(u * C + j) * L + i]);
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = float32_to_bfloat16(bfloat16_to_float32(px[o]) * powf(pdat->bias + over * sum, -pdat->beta));
			}
		}
	}
}

static void LRN_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint16_t* py = (uint16_t*)y->datas;
	float sum, t;
	float over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for (u = 0; u < N; u++) {
		for (v = 0; v < C; v++) {
			for (i = 0; i < L; i++) {
				start = v - (pdat->size / 2);
				if (start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if (end >= C)
					end = C - 1;
				for (j = start, sum = 0; j <= end; ++j) {
					t = float16_to_float32(px[(u * C + j) * L + i]);
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = float32_to_float16(float16_to_float32(px[o]) * powf(pdat->bias + over * sum, -pdat->beta));
			}
		}
	}
}

template <typename T>
static void LRN_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	T sum, t;
	T over = pdat->alpha / pdat->size;
	int N = x->dims[0];
	int C = x->dims[1];
	int L = x->strides[1];
	int start, end;
	int i, j, u, v, o;

	for (u = 0; u < N; u++) {
		for (v = 0; v < C; v++) {
			for (i = 0; i < L; i++) {
				start = v - (pdat->size / 2);
				if (start < 0)
					start = 0;
				end = v + (pdat->size / 2);
				if (end >= C)
					end = C - 1;
				for (j = start, sum = 0; j <= end; ++j) {
					t = px[(u * C + j) * L + i];
					sum += t * t;
				}
				o = (u * C + v) * L + i;
				py[o] = px[o] * pow(pdat->bias + over * sum, -pdat->beta);
			}
		}
	}
}

void resolver_default_op_LRN(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = LRN_bfloat16,
			.float16_ = LRN_float16,
			.float32_ = LRN_generic<float>,
			.float64_ = LRN_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = LRN_float16,
			.float32_ = LRN_generic<float>,
			.float64_ = LRN_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LRN_init;
		n->exit = LRN_exit;
		n->reshape = LRN_reshape;
	}
}
