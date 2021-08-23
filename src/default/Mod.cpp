#include <onnx.h>
#include "util.h"

namespace {

struct operator_pdata_t {
	int fmod;
};

bool Mod_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 2, 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->fmod = n->attribute_read_int("fmod", 0);
	n->priv = pdat;
	return true;
}

int Mod_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int Mod_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
void Mod_generic(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
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

void resolver_default_op_Mod(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Mod_init;
		n->exit = Mod_exit;
		n->reshape = Mod_reshape;
	}
}
