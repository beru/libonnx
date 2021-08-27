#include <onnx.h>
#include "util.h"

namespace {

bool Where_init(onnx_node_t* n)
{
	return is_inout_size(n, 3, 1);
}

int Where_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[n->inputs.size() - 1]))
		return 0;
	for (i = n->inputs.size() - 2; i >= 0; i--) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Where_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x0 = n->inputs[0];
	onnx_tensor_t* x1 = n->inputs[1];
	onnx_tensor_t* x2 = n->inputs[2];
	T* py = (T*)y->data;
	T* px;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		uint8_t* c = (uint8_t*)x0->broadcast_map_address(y, i);
		if (*c)
			px = (T*)x1->broadcast_map_address(y, i);
		else
			px = (T*)x2->broadcast_map_address(y, i);
		py[i] = *px;
	}
}

GEN_HOLEDR_TYPE(holder, Where_generic)

} // namespace

void resolver_default_op_Where(onnx_node_t* n)
{
	if (n->opset >= 9) {
		if (n->inputs.size() == 3) {
			n->ope = onnx_ope_type_select<holder,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::complex<float>, std::complex<double>,
				std::string
			>(n->inputs[0]->type);
		}
	}
	if (n->ope) {
		n->init = Where_init;
		n->reshape = Where_reshape;
	}
}
