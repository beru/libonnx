#include <onnx.h>

static int Mean_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Mean_exit(onnx_node_t* n)
{
	return 1;
}

static int Mean_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[0], n->inputs[0]->type))
		return 0;
	for (i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

static void Mean_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float sum;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, sum = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			sum += bfloat16_to_float32(*px);
		}
		py[i] = float32_to_bfloat16(sum / n->inputs.size());
	}
}

static void Mean_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	float sum;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, sum = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (uint16_t*)x->broadcast_map_address(y, i);
			sum += float16_to_float32(*px);
		}
		py[i] = float32_to_float16(sum / n->inputs.size());
	}
}

static void Mean_float32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	float* py = (float*)y->datas;
	float* px;
	float sum;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, sum = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (float*)x->broadcast_map_address(y, i);
			sum += *px;
		}
		py[i] = sum / n->inputs.size();
	}
}

static void Mean_float64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	double* py = (double*)y->datas;
	double* px;
	double sum;
	size_t i, j, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, sum = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (double*)x->broadcast_map_address(y, i);
			sum += *px;
		}
		py[i] = sum / n->inputs.size();
	}
}

void resolver_default_op_Mean(onnx_node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 8) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 6) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float64;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->ope = Mean_float64;
			break;
		default:
			break;
		}
	}
}
