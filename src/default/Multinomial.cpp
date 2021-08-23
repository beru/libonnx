#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	int sample_size;
	float seed;
};

bool Multinomial_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 6);
	pdat->sample_size = n->attribute_read_int("sample_size", 1);
	pdat->seed = n->attribute_read_float("seed", 0.0);
	n->priv = pdat;
	return true;
}

int Multinomial_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int Multinomial_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, pdat->dtype);
}

template <typename T>
void Multinomial_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	T* px = (T*)x->data;
	std::vector<T> cum(csz);
	int i, j, k, l, o;

	if (pdat->seed != 0.0)
		srand(pdat->seed);

	switch (y->type) {
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((T)rand() / (T)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((T)rand() / (T)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

GEN_HOLEDR_TYPE(holder, Multinomial_generic)

} // namespace

void resolver_default_op_Multinomial(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Multinomial_init;
		n->exit = Multinomial_exit;
		n->reshape = Multinomial_reshape;
	}
}
