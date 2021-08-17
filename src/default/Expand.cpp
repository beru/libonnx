#include <onnx.h>

static int Expand_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Expand_exit(onnx_node_t* n)
{
	return 1;
}

static int Expand_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* s = n->inputs[1];
	int64_t* ps = (int64_t*)s->datas;
	int ndim = max(x->ndim, (int)s->ndata);
	std::vector<int> dims(ndim);
	int i, j, k;

	for (i = x->ndim - 1, j = s->ndata - 1, k = ndim - 1; k >= 0; k--) {
		if (i < 0)
			dims[k] = ps[j--];
		else if (j < 0)
			dims[k] = x->dims[i--];
		else {
			if (x->dims[i] == ps[j])
				dims[k] = x->dims[i];
			else if ((x->dims[i] == 1) || (ps[j] == 1))
				dims[k] = (x->dims[i] > ps[j]) ? x->dims[i] : ps[j];
			else
				return 0;
			i--;
			j--;
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
static void Expand_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	T* py = (T*)y->datas;
	T* px = (T*)x->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (T*)x->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Expand_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px = (uint16_t*)x->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (uint16_t*)x->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Expand_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* px = (uint16_t*)x->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (uint16_t*)x->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

static void Expand_complex64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	float* py = (float*)y->datas;
	float* px = (float*)x->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (float*)x->broadcast_map_address(y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Expand_complex128(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	double* py = (double*)y->datas;
	double* px = (double*)x->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (double*)x->broadcast_map_address(y, i);
		py[i * 2] = px[0];
		py[i * 2 + 1] = px[1];
	}
}

static void Expand_string(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	char** px = (char**)x->datas;
	char** py = (char**)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (char**)x->broadcast_map_address(y, i);
		if (py[i])
			free(py[i]);
		py[i] = strdup(px[i]);
	}
}

void resolver_default_op_Expand(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Expand_generic<uint8_t>,
			.int8_ = Expand_generic<int8_t>,
			.int16_ = Expand_generic<int16_t>,
			.int32_ = Expand_generic<int32_t>,
			.int64_ = Expand_generic<int64_t>,
			.uint8_ = Expand_generic<uint8_t>,
			.uint16_ = Expand_generic<uint16_t>,
			.uint32_ = Expand_generic<uint32_t>,
			.uint64_ = Expand_generic<uint64_t>,
			.bfloat16_ = Expand_bfloat16,
			.float16_ = Expand_float16,
			.float32_ = Expand_generic<float>,
			.float64_ = Expand_generic<double>,
			.complex64_ = Expand_complex64,
			.complex128_ = Expand_complex128,
			.string_ = Expand_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Expand_generic<uint8_t>,
			.int8_ = Expand_generic<int8_t>,
			.int16_ = Expand_generic<int16_t>,
			.int32_ = Expand_generic<int32_t>,
			.int64_ = Expand_generic<int64_t>,
			.uint8_ = Expand_generic<uint8_t>,
			.uint16_ = Expand_generic<uint16_t>,
			.uint32_ = Expand_generic<uint32_t>,
			.uint64_ = Expand_generic<uint64_t>,
			.float16_ = Expand_float16,
			.float32_ = Expand_generic<float>,
			.float64_ = Expand_generic<double>,
			.complex64_ = Expand_complex64,
			.complex128_ = Expand_complex128,
			.string_ = Expand_string,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Expand_init;
		n->exit = Expand_exit;
		n->reshape = Expand_reshape;
	}
}
