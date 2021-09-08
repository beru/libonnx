#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	int fmod;
};

bool Mod_init(node_t* n)
{
	if (!is_inout_size(n, 2, 1)) {
		return false;
	}
	auto pdat = std::make_shared<operator_pdata_t>();
	if (!pdat)
		return false;
	pdat->fmod = n->attribute("fmod", 0);
	n->priv = pdat;
	return true;
}

int Mod_reshape(node_t* n)
{
	tensor_t* y = n->outputs[0];
	tensor_t* a = n->inputs[0];
	tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
void Mod_generic(node_t* n)
{
	auto pdat = std::static_pointer_cast<operator_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	tensor_t* a = n->inputs[0];
	tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;

	if (pdat->fmod) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T* pa = (T*)a->broadcast_map_address(y, i);
			T* pb = (T*)b->broadcast_map_address(y, i);
			py[i] = fmod(*pa, *pb);
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T* pa = (T*)a->broadcast_map_address(y, i);
			T* pb = (T*)b->broadcast_map_address(y, i);
			T t;
			if constexpr (std::is_integral_v<T>) {
				t = *pa % *pb;
				if (((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0)))
					t += *pb;
			}else {
				t = fmod(*pa, *pb);
			}
			py[i] = t;
		}
	}
}

GEN_HOLEDR_TYPE(holder, Mod_generic)

} // namespace

void resolver_default_op_Mod(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Mod_init;
		n->reshape = Mod_reshape;
	}
}

} // namespace onnx
