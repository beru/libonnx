#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Clip_operator : public operator_t {
	void* pmin = nullptr;
	void* pmax = nullptr;

	bool init() override {
		if (!(inputs.size() >= 1 && outputs.size() == 1)) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		pmin = nullptr;
		pmax = nullptr;
		for (size_t i = 1; i < min<size_t>(3, inputs.size()); i++) {
			if (inputs[i]->ndim == 0) {
				if (inputs[i]->name == "min")
					pmin = inputs[i]->data;
				else if (inputs[i]->name == "max")
					pmax = inputs[i]->data;
			}
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T minv = pmin ? *(T*)pmin : std::numeric_limits<T>::lowest();
		T maxv = pmax ? *(T*)pmax : std::numeric_limits<T>::max();
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T v = px[i];
			if (v < minv)
				v = minv;
			else if (v > maxv)
				v = maxv;
			py[i] = v;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 12) {
			TYPED_EXEC(type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 11) {
			TYPED_EXEC(type,
				float16_t, float, double
			)
		}else if (opset >= 6) {
		}else if (opset >= 1) {
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Clip(int opset)
{
	return new Clip_operator;
}

} // namespace onnx
