#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct PRelu_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	template <typename T>
	bool exec() {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		T* py = (T*)y->data;
		const T* pa = (const T*)a->data;;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			T va = pa[i];
			if (va < 0) {
				const T* pb = (const T*)b->broadcast_map_address(y, i);
				va *= (*pb);
			}
			py[i] = va;
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 9) {
			return typed_exec<PRelu_operator,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			return typed_exec<PRelu_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 6) {
			return typed_exec<PRelu_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			return typed_exec<PRelu_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_PRelu(int opset) { return new (std::nothrow) PRelu_operator; }

} // namespace onnx
