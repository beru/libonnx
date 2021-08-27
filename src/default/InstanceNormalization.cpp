#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t : public onnx_node_t::ope_pdata_t {
	float epsilon;
};

bool InstanceNormalization_init(onnx_node_t* n)
{
	if (!(n->inputs.size() == 3 && n->outputs.size() >= 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->epsilon = n->attribute_read_float("epsilon", 1e-05);
	n->priv = pdat;
	return true;
}

int InstanceNormalization_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void InstanceNormalization_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* pscale = (T*)scale->data;
	T* pb = (T*)b->data;
	T* py = (T*)y->data;
	T temp, mean, var;
	int N = x->dims[0];
	int C = x->dims[1];
	int NC = N * C;
	int channel = 1;
	int i, j, l, o, jc;

	for (i = 2; i < x->ndim; i++)
		channel *= x->dims[i];
	for (j = 0; j < NC; j++) {
		o = j * channel;
		l = o + channel;
		jc = j % C;
		temp = 0;
		for (i = o; i < l; i++)
			temp += px[i];
		mean = temp / channel;
		temp = 0;
		for (i = o; i < l; i++)
			temp += pow(px[i] - mean, 2);
		var = temp / channel;
		double denom = sqrt((double)var + pdat->epsilon);
		double tmp = pb[jc];
		for (i = o; i < l; i++) {
			py[i] = pscale[jc] * ((px[i] - mean) / denom) + tmp;
		}
	}
}

GEN_HOLEDR_TYPE(holder, InstanceNormalization_generic)

} // namespace

void resolver_default_op_InstanceNormalization(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = InstanceNormalization_init;
		n->reshape = InstanceNormalization_reshape;
	}
}
