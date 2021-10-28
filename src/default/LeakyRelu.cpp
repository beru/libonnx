#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct LeakyRelu_operator : public operator_t {
	float alpha;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		alpha = attribute("alpha", 0.01f);
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			T v = px[i];
			if (v < 0) {
				v *= alpha;
			}
			py[i] = v;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 6) {
			typed_exec<LeakyRelu_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<LeakyRelu_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_LeakyRelu(int opset) { return new (std::nothrow) LeakyRelu_operator; }

} // namespace onnx
