#include <onnx.h>

static int Where_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 3) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Where_exit(onnx_node_t* n)
{
	return 1;
}

static int Where_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[n->inputs.size() - 1], n->inputs[n->inputs.size() - 1]->type))
		return 0;
	for (i = n->inputs.size() - 2; i >= 0; i--) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Where_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	T* py = (T*)y->datas;
	T* px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (T*)x1->broadcast_map_address(y, i);
		else
			px = (T*)x2->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Where_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (uint16_t*)x1->broadcast_map_address(y, i);
		else
			px = (uint16_t*)x2->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Where_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (uint16_t*)x1->broadcast_map_address(y, i);
		else
			px = (uint16_t*)x2->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Where_complex64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	float* py = (float*)y->datas;
	float* px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (float*)x1->broadcast_map_address(y, i);
		else
			px = (float*)x2->broadcast_map_address(y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_complex128(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	double* py = (double*)y->datas;
	double* px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (double*)x1->broadcast_map_address(y, i);
		else
			px = (double*)x2->broadcast_map_address(y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Where_string(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	char** py = (char**)y->datas;
	char** px;
	uint8_t* c;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (char**)x1->broadcast_map_address(y, i);
		else
			px = (char**)x2->broadcast_map_address(y, i);
		if (py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Where(onnx_node_t* n)
{
	if (n->opset >= 9) {
		if (n->inputs.size() == 3) {
			switch (n->inputs[2]->type)	{
			case ONNX_TENSOR_TYPE_BOOL:
				n->ope = Where_generic<uint8_t>;
				break;
			case ONNX_TENSOR_TYPE_INT8:
				n->ope = Where_generic<int8_t>;
				break;
			case ONNX_TENSOR_TYPE_INT16:
				n->ope = Where_generic<int16_t>;
				break;
			case ONNX_TENSOR_TYPE_INT32:
				n->ope = Where_generic<int32_t>;
				break;
			case ONNX_TENSOR_TYPE_INT64:
				n->ope = Where_generic<int64_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT8:
				n->ope = Where_generic<uint8_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT16:
				n->ope = Where_generic<uint16_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT32:
				n->ope = Where_generic<uint32_t>;
				break;
			case ONNX_TENSOR_TYPE_UINT64:
				n->ope = Where_generic<uint64_t>;
				break;
			case ONNX_TENSOR_TYPE_BFLOAT16:
				n->ope = Where_bfloat16;
				break;
			case ONNX_TENSOR_TYPE_FLOAT16:
				n->ope = Where_float16;
				break;
			case ONNX_TENSOR_TYPE_FLOAT32:
				n->ope = Where_generic<float>;
				break;
			case ONNX_TENSOR_TYPE_FLOAT64:
				n->ope = Where_generic<double>;
				break;
			case ONNX_TENSOR_TYPE_COMPLEX64:
				n->ope = Where_complex64;
				break;
			case ONNX_TENSOR_TYPE_COMPLEX128:
				n->ope = Where_complex128;
				break;
			case ONNX_TENSOR_TYPE_STRING:
				n->ope = Where_string;
				break;
			default:
				break;
			}
		}
	}
	if (n->ope) {
		n->init = Where_init;
		n->exit = Where_exit;
		n->reshape = Where_reshape;
	}
}
