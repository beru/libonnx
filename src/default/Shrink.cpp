#include <onnx.h>
#include "float16.h"

struct ope_pdata_t {
	float bias;
	float lambd;
};

static int Shrink_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->bias = n->attribute_read_float("bias", 0.0);
		pdat->lambd = n->attribute_read_float("lambd", 0.5);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Shrink_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Shrink_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Shrink_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < -pdat->lambd)
			py[i] = px[i] + (T)pdat->bias;
		else if (px[i] > pdat->lambd)
			py[i] = px[i] - (T)pdat->bias;
		else
			py[i] = 0;
	}
}

void resolver_default_op_Shrink(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Shrink_generic<int8_t>,
			.int16_ = Shrink_generic<int16_t>,
			.int32_ = Shrink_generic<int32_t>,
			.int64_ = Shrink_generic<int64_t>,
			.uint8_ = Shrink_generic<uint8_t>,
			.uint16_ = Shrink_generic<uint16_t>,
			.uint32_ = Shrink_generic<uint32_t>,
			.uint64_ = Shrink_generic<uint64_t>,
			.float16_ = Shrink_generic<float16_t>,
			.float32_ = Shrink_generic<float>,
			.float64_ = Shrink_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Shrink_init;
		n->exit = Shrink_exit;
		n->reshape = Shrink_reshape;
	}
}
