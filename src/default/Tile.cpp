#include <complex>
#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Tile_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Tile_exit(onnx_node_t* n)
{
	return 1;
}

static int Tile_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* r = n->inputs[1];
	int64_t* pr = (int64_t*)r->datas;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int i;

	for (i = 0; i < ndim; i++)
		dims[i] = x->dims[i] * pr[i];
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
static void Tile_generic(onnx_node_t* n)
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

static void Tile_string(onnx_node_t* n)
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

void resolver_default_op_Tile(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Tile_generic<uint8_t>,
			.int8_ = Tile_generic<int8_t>,
			.int16_ = Tile_generic<int16_t>,
			.int32_ = Tile_generic<int32_t>,
			.int64_ = Tile_generic<int64_t>,
			.uint8_ = Tile_generic<uint8_t>,
			.uint16_ = Tile_generic<uint16_t>,
			.uint32_ = Tile_generic<uint32_t>,
			.uint64_ = Tile_generic<uint64_t>,
			.bfloat16_ = Tile_generic<bfloat16_t>,
			.float16_ = Tile_generic<float16_t>,
			.float32_ = Tile_generic<float>,
			.float64_ = Tile_generic<double>,
			.complex64_ = Tile_generic<std::complex<float>>,
			.complex128_ = Tile_generic<std::complex<double>>,
			.string_ = Tile_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Tile_generic<uint8_t>,
			.int8_ = Tile_generic<int8_t>,
			.int16_ = Tile_generic<int16_t>,
			.int32_ = Tile_generic<int32_t>,
			.int64_ = Tile_generic<int64_t>,
			.uint8_ = Tile_generic<uint8_t>,
			.uint16_ = Tile_generic<uint16_t>,
			.uint32_ = Tile_generic<uint32_t>,
			.uint64_ = Tile_generic<uint64_t>,
			.float16_ = Tile_generic<float16_t>,
			.float32_ = Tile_generic<float>,
			.float64_ = Tile_generic<double>,
			.complex64_ = Tile_generic<std::complex<float>>,
			.complex128_ = Tile_generic<std::complex<double>>,
			.string_ = Tile_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Tile_init;
		n->exit = Tile_exit;
		n->reshape = Tile_reshape;
	}
}
