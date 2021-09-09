#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	tensor_type_t dtype;
	int sample_size;
	float seed;
};

bool Multinomial_init(node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	pdat->dtype = (tensor_type_t)n->attribute("dtype", ONNX_TENSOR_TYPE_INT32);
	pdat->sample_size = n->attribute("sample_size", 1);
	pdat->seed = n->attribute("seed", 0.0f);
	n->priv = pdat;
	return true;
}

int Multinomial_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, pdat->dtype);
}

template <typename XT, typename YT>
void Multinomial_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	const XT* px = (const XT*)x->data;
	std::vector<XT> cum(csz);

	if (pdat->seed != 0.0)
		srand(pdat->seed);

	YT* py = (YT*)y->data;
	for (int i = 0; i < bsz; i++) {
		for (int j = 0; j < pdat->sample_size; j++) {
			cum[0] = px[i * csz];
			for (int k = 1; k < csz; k++)
				cum[k] = cum[k - 1] + px[i * csz + k];
			int l = csz - 1;
			for (int k = 0; k < csz - 1; k++) {
				if ((XT)rand() / (XT)(RAND_MAX) < cum[k]) {
					l = k;
					break;
				}
			}
			int o = i * csz + l;
			py[o]++;
		}
	}
}

template <typename T>
void Multinomial_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	switch (y->type) {
	case ONNX_TENSOR_TYPE_INT32:
		Multinomial_generic<T, int32_t>(n);
		break;
	case ONNX_TENSOR_TYPE_INT64:
		Multinomial_generic<T, int64_t>(n);
		break;
	default:
		break;
	}
}

GEN_HOLEDR_TYPE(holder, Multinomial_generic)

} // namespace

void resolver_default_op_Multinomial(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Multinomial_init;
		n->reshape = Multinomial_reshape;
	}
}

} // namespace onnx
