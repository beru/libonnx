#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Where_operator : public operator_t {

	bool init() override {
		return is_inout_size(3, 1);
	}

	bool reshape() override {
		const tensor_t* condition = inputs[0];
		const tensor_t* x = inputs[1];
		const tensor_t* y = inputs[2];
		tensor_t* output = outputs[0];
		if (condition->type != ONNX_TENSOR_TYPE_BOOL) {
			return false;
		}
		if (x->type != y->type || x->type != output->type) {
			return false;
		}

		if (!output->reshape_identity(inputs[inputs.size() - 1])) {
			return false;
		}
		for (int i = inputs.size() - 2; i >= 0; i--) {
			if (!output->reshape_multi_broadcast(output, inputs[i], output->type))
				return false;
		}
		return true;
	}

	template <typename T>
	bool exec() {
		const tensor_t* condition = inputs[0];
		const tensor_t* x = inputs[1];
		const tensor_t* y = inputs[2];
		tensor_t* output = outputs[0];
		T* pout = (T*)output->data;
		T* pin;

		for (size_t i = 0, l = output->ndata; i < l; ++i) {
			bool_t* c = (bool_t*)condition->broadcast_map_address(output, i);
			if (*c) {
				pin = (T*)x->broadcast_map_address(output, i);
			}else {
				pin = (T*)y->broadcast_map_address(output, i);
			}
			pout[i] = *pin;
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[1]->type;
		if (inputs.size() != 3) {
			return false;
		}
		if (opset >= 16) {
			return typed_exec<Where_operator,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}else if (opset >= 9) {
			return typed_exec<Where_operator,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}
		return false;
	}
};

} // namespace {

operator_t* resolver_default_op_Where(int opset)
{
	return new (std::nothrow) Where_operator;
}

} // namespace onnx
