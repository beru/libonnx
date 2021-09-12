#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Mod_operator : public operator_t {
	int attr_fmod;

	bool init() override {
		if (!is_inout_size(2, 1)) {
			return false;
		}
		attr_fmod = n->attribute("fmod", 0);
		return true;
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		return y->reshape_multi_broadcast(a, b, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		T* py = (T*)y->data;

		if (attr_fmod) {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				const T* pa = (const T*)a->broadcast_map_address(y, i);
				const T* pb = (const T*)b->broadcast_map_address(y, i);
				py[i] = fmod(*pa, *pb);
			}
		}else {
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				const T* pa = (const T*)a->broadcast_map_address(y, i);
				const T* pb = (const T*)b->broadcast_map_address(y, i);
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

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 10) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_Mod(node_t* n)
{
	n->ope = std::make_shared<Mod_operator>();
}

} // namespace onnx
