#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	float alpha;
	float beta;
	int transA;
	int transB;

	int m;
	int n;
	int k;
};

bool Gemm_init(node_t* n)
{
	if (!(n->inputs.size() >= 2 && n->outputs.size() == 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	if (!pdat)
		return false;
	pdat->alpha = n->attribute("alpha", 1.0f);
	pdat->beta = n->attribute("beta", 1.0f);
	pdat->transA = n->attribute("transA", 0);
	pdat->transB = n->attribute("transB", 0);
	pdat->m = 0;
	pdat->n = 0;
	pdat->k = 0;
	n->priv = pdat;
	return true;
}

int Gemm_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	tensor_t* a = n->inputs[0];
	tensor_t* b = n->inputs[1];
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
void Gemm_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	tensor_t* a = n->inputs[0];
	tensor_t* b = n->inputs[1];
	tensor_t* c = (n->inputs.size() > 2) ? n->inputs[2] : nullptr;
	T* py = (T*)y->data;
	T* pa = (T*)a->data;
	T* pb = (T*)b->data;
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

GEN_HOLEDR_TYPE(holder, Gemm_generic)

} // namespace

void resolver_default_op_Gemm(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Gemm_init;
		n->reshape = Gemm_reshape;
	}
}

} // namespace onnx
