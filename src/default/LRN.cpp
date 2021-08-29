#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	float alpha;
	float beta;
	float bias;
	int size;
};

bool LRN_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->alpha = n->attribute_read_float("alpha", 0.0001);
	pdat->beta = n->attribute_read_float("beta", 0.75);
	pdat->bias = n->attribute_read_float("bias", 1.0);
	pdat->size = n->attribute_read_int("size", 1);
	n->priv = pdat;
	return true;
}

template <typename T>
void LRN_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;
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

GEN_HOLEDR_TYPE(holder, LRN_generic)

} // namespace

void resolver_default_op_LRN(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LRN_init;
	}
}
