#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	bool isleft;
};

bool BitShift_init(node_t* n)
{
	if (!is_inout_size(n, 2, 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	pdat->isleft = (strcmp(n->attribute("direction", "LEFT"), "LEFT") == 0);
	n->priv = pdat;
	return true;
}

int BitShift_reshape(node_t* n)
{
	tensor_t* y = n->outputs[0];
	const tensor_t* a = n->inputs[0];
	const tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
void BitShift_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	const tensor_t* a = n->inputs[0];
	const tensor_t* b = n->inputs[1];
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

void resolver_default_op_BitShift(node_t* n)
{
	if (n->opset >= 11) {
		n->ope = ope_type_select<holder,
			uint8_t, uint16_t, uint32_t, uint64_t
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = BitShift_init;
		n->reshape = BitShift_reshape;
	}
}

} // namespace onnx
