#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

struct ope_pdata_t {
	float alpha;
	float beta;
	int transA;
	int transB;

	int m;
	int n;
	int k;
};

static int Gemm_init(onnx_node_t* n)
{

	if ((n->inputs.size() >= 2) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->alpha = n->attribute_read_float("alpha", 1.0);
		pdat->beta = n->attribute_read_float("beta", 1.0);
		pdat->transA = n->attribute_read_int("transA", 0);
		pdat->transB = n->attribute_read_int("transB", 0);
		pdat->m = 0;
		pdat->n = 0;
		pdat->k = 0;
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Gemm_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Gemm_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int k;

	if (pdat->transA) {
		pdat->m = a->dims[1];
		pdat->k = a->dims[0];
	}else {
		pdat->m = a->dims[0];
		pdat->k = a->dims[1];
	}
	if (pdat->transB) {
		pdat->n = b->dims[0];
		k = 1;
	}else {
		pdat->n = b->dims[1];
		k = 0;
	}
	if (b->dims[k] != pdat->k)
		return 0;
	if (pdat->m <= 0 || pdat->n <= 0 || pdat->k <= 0)
		return 0;
	int tmp[2] = { pdat->m, pdat->n };
	if ((n->inputs.size() > 2) && !n->inputs[2]->broadcast_is_valid(tmp, 2))
		return 0;
	return y->reshape(tmp, 2, a->type);
}

template <typename T>
static void Gemm_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	onnx_tensor_t* c = (n->inputs.size() > 2) ? n->inputs[2] : NULL;
	T* py = (T*)y->datas;
	T* pa = (T*)a->datas;
	T* pb = (T*)b->datas;
	T* pc;
	T sum;
	int oa = 0;
	int ob = 0;
	int oy = 0;
	int i, j, k;

	if (pdat->transA && pdat->transB) {
		for (i = 0; i < pdat->m; i++) {
			for (j = 0; j < pdat->n; j++) {
				sum = 0;
				for (k = 0; k < pdat->k; k++) {
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += 1;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->k;
				if (c) {
					pc = (T*)c->broadcast_map_address(y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa++;
		}
	}else if (pdat->transA) {
		for (i = 0; i < pdat->m; i++) {
			for (j = 0; j < pdat->n; j++) {
				sum = 0;
				for (k = 0; k < pdat->k; k++) {
					sum += pa[oa] * pb[ob];
					oa += pdat->m;
					ob += pdat->n;
				}
				oa -= pdat->m * pdat->k;
				ob -= pdat->n * pdat->k;
				if (c) {
					pc = (T*)c->broadcast_map_address(y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa++;
		}
	}else if (pdat->transB) {
		for (i = 0; i < pdat->m; i++) {
			for (j = 0; j < pdat->n; j++) {
				sum = 0;
				for (k = 0; k < pdat->k; k++) {
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += 1;
				}
				oa -= pdat->k;
				ob -= pdat->k;
				if (c) {
					pc = (T*)c->broadcast_map_address(y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob += pdat->k;
			}
			ob -= pdat->n * pdat->k;
			oa += pdat->k;
		}
	}else {
		for (i = 0; i < pdat->m; i++) {
			for (j = 0; j < pdat->n; j++) {
				sum = 0;
				for (k = 0; k < pdat->k; k++) {
					sum += pa[oa] * pb[ob];
					oa += 1;
					ob += pdat->n;
				}
				oa -= pdat->k;
				ob -= pdat->n * pdat->k;
				if (c) {
					pc = (T*)c->broadcast_map_address(y, oy);
					py[oy] = pdat->alpha * sum + pdat->beta * (*pc);
				}
				else
					py[oy] = pdat->alpha * sum;
				oy++;
				ob++;
			}
			ob -= pdat->n;
			oa += pdat->k;
		}
	}
}

void resolver_default_op_Gemm(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Gemm_generic<int32_t>,
			.int64_ = Gemm_generic<int64_t>,
			.uint32_ = Gemm_generic<uint32_t>,
			.uint64_ = Gemm_generic<uint64_t>,
			.bfloat16_ = Gemm_generic<bfloat16_t>,
			.float16_ = Gemm_generic<float16_t>,
			.float32_ = Gemm_generic<float>,
			.float64_ = Gemm_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Gemm_generic<int32_t>,
			.int64_ = Gemm_generic<int64_t>,
			.uint32_ = Gemm_generic<uint32_t>,
			.uint64_ = Gemm_generic<uint64_t>,
			.float16_ = Gemm_generic<float16_t>,
			.float32_ = Gemm_generic<float>,
			.float64_ = Gemm_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Gemm_generic<int32_t>,
			.int64_ = Gemm_generic<int64_t>,
			.uint32_ = Gemm_generic<uint32_t>,
			.uint64_ = Gemm_generic<uint64_t>,
			.float16_ = Gemm_generic<float16_t>,
			.float32_ = Gemm_generic<float>,
			.float64_ = Gemm_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Gemm_generic<float16_t>,
			.float32_ = Gemm_generic<float>,
			.float64_ = Gemm_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Gemm_init;
		n->exit = Gemm_exit;
		n->reshape = Gemm_reshape;
	}
}
