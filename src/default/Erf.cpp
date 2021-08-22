#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Erf_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Erf_exit(onnx_node_t* n)
{
	return 1;
}

static int Erf_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Erf_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	int i, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		py[i] = erf(px[i]);
	}
}

void resolver_default_op_Erf(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Erf_generic<int8_t>,
			.int16_ = Erf_generic<int16_t>,
			.int32_ = Erf_generic<int32_t>,
			.int64_ = Erf_generic<int64_t>,
			.uint8_ = Erf_generic<uint8_t>,
			.uint16_ = Erf_generic<uint16_t>,
			.uint32_ = Erf_generic<uint32_t>,
			.uint64_ = Erf_generic<uint64_t>,
			.bfloat16_ = Erf_generic<bfloat16_t>,
			.float16_ = Erf_generic<float16_t>,
			.float32_ = Erf_generic<float>,
			.float64_ = Erf_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Erf_generic<int8_t>,
			.int16_ = Erf_generic<int16_t>,
			.int32_ = Erf_generic<int32_t>,
			.int64_ = Erf_generic<int64_t>,
			.uint8_ = Erf_generic<uint8_t>,
			.uint16_ = Erf_generic<uint16_t>,
			.uint32_ = Erf_generic<uint32_t>,
			.uint64_ = Erf_generic<uint64_t>,
			.float16_ = Erf_generic<float16_t>,
			.float32_ = Erf_generic<float>,
			.float64_ = Erf_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Erf_init;
		n->exit = Erf_exit;
		n->reshape = Erf_reshape;
	}
}
