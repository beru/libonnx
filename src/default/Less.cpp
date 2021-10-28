#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Less_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			const T* pa = (const T*)a->broadcast_map_address(y, i);
			const T* pb = (const T*)b->broadcast_map_address(y, i);
			py[i] = (*pa < *pb);
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Less_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 9) {
			typed_exec<Less_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			typed_exec<Less_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Less(int opset) { return new (std::nothrow) Less_operator; }

} // namespace onnx
