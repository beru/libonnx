#include <onnx.h>

static int IsNaN_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int IsNaN_exit(onnx_node_t* n)
{
	return 1;
}

static int IsNaN_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
}

static void IsNaN_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = bfloat16_to_float32(px[i]);
		py[i] = isnan(v) ? 1 : 0;
	}
}

static void IsNaN_float16(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		py[i] = isnan(v) ? 1 : 0;
	}
}

static void IsNaN_float32(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = isnan(px[i]) ? 1 : 0;
}

static void IsNaN_float64(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->datas;
	uint8_t* py = (uint8_t*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = isnan(px[i]) ? 1 : 0;
}

void resolver_default_op_IsNaN(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->ope = IsNaN_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = IsNaN_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = IsNaN_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = IsNaN_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 9) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = IsNaN_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = IsNaN_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = IsNaN_float64;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = IsNaN_init;
		n->exit = IsNaN_exit;
		n->reshape = IsNaN_reshape;
	}
}
