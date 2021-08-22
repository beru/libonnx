#include <onnx.h>
#include "float16.h"

struct operator_pdata_t {
	float epsilon;
};

static int InstanceNormalization_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 3) && (n->outputs.size() >= 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->epsilon = n->attribute_read_float("epsilon", 1e-05);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int InstanceNormalization_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int InstanceNormalization_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void InstanceNormalization_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* scale = n->inputs[1];
	onnx_tensor_t* b = n->inputs[2];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* pscale = (T*)scale->datas;
	T* pb = (T*)b->datas;
	T* py = (T*)y->datas;
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

void resolver_default_op_InstanceNormalization(onnx_node_t* n)
{
	if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = InstanceNormalization_generic<float16_t>,
			.float32_ = InstanceNormalization_generic<float>,
			.float64_ = InstanceNormalization_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = InstanceNormalization_generic<float16_t>,
			.float32_ = InstanceNormalization_generic<float>,
			.float64_ = InstanceNormalization_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = InstanceNormalization_init;
		n->exit = InstanceNormalization_exit;
		n->reshape = InstanceNormalization_reshape;
	}
}
