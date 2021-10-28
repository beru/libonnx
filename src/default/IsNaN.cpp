#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct IsNaN_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
	}

	template <typename T>
	bool exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			py[i] = isnan(px[i]);
		}
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			return typed_exec<IsNaN_operator,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 9) {
			return typed_exec<IsNaN_operator,
				float16_t, float, double
			>(this, type);
		}else {
			return false;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_IsNaN(int opset) { return new (std::nothrow) IsNaN_operator; }

} // namespace onnx
