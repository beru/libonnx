#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	float alpha;
	float beta;
	float bias;
	int size;
};

bool LRN_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	pdat->alpha = n->attribute("alpha", 0.0001f);
	pdat->beta = n->attribute("beta", 0.75f);
	pdat->bias = n->attribute("bias", 1.0f);
	pdat->size = n->attribute("size", 1);
	n->priv = pdat;
	return true;
}

template <typename T>
void LRN_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
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

void resolver_default_op_LRN(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LRN_init;
	}
}

} // namespace onnx
