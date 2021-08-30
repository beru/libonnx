#include <onnx.h>
#include "util.h"

namespace {

int Tile_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* r = n->inputs[1];
	int64_t* pr = (int64_t*)r->data;
	int ndim = x->ndim;
	std::vector<int> dims(ndim);

	for (int i = 0; i < ndim; i++)
		dims[i] = x->dims[i] * pr[i];
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
void Tile_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	T* py = (T*)y->data;
	T* px = (T*)x->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		px = (T*)x->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

GEN_HOLEDR_TYPE(holder, Tile_generic)

} // namespace

void resolver_default_op_Tile(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bool_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double, bfloat16_t,
			std::complex<float>, std::complex<double>,
			std::string
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			bool_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double,
			std::complex<float>, std::complex<double>,
			std::string
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = [](onnx_node_t* n){
			return is_inout_size(n, 2, 1);
		};
		n->reshape = Tile_reshape;
	}
}
