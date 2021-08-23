#include <onnx.h>
#include "util.h"

namespace {

struct ope_pdata_t {
	int isleft;
};

bool BitShift_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 2, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->isleft = (strcmp(n->attribute_read_string("direction", "LEFT"), "LEFT") == 0) ? 1 : 0;
	n->priv = pdat;
	return true;
}

int BitShift_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int BitShift_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
void BitShift_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;

	if (pdat->isleft) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T* pa = (T*)a->broadcast_map_address(y, i);
			T* pb = (T*)b->broadcast_map_address(y, i);
			py[i] = *pa << *pb;
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T* pa = (T*)a->broadcast_map_address(y, i);
			T* pb = (T*)b->broadcast_map_address(y, i);
			py[i] = *pa >> *pb;
		}
	}
}

GEN_HOLEDR_TYPE(holder, BitShift_generic)

} // namespace

void resolver_default_op_BitShift(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint16_t, uint32_t, uint64_t
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->reshape = BitShift_reshape;
	}
}
