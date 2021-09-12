#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Clip_operator : public operator_t {
	void* pmin = nullptr;
	void* pmax = nullptr;

	bool init() override {
		if (!(n->inputs.size() >= 1 && n->outputs.size() == 1)) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		pmin = nullptr;
		pmax = nullptr;
		for (int i = 1; i < min<size_t>(3, n->inputs.size()); i++) {
			if (n->inputs[i]->ndim == 0) {
				if (n->inputs[i]->name == "min")
					pmin = n->inputs[i]->data;
				else if (n->inputs[i]->name == "max")
					pmax = n->inputs[i]->data;
			}
		}
		return y->reshape_identity(x);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T minv = pmin ? *(T*)pmin : std::numeric_limits<T>::lowest();
		T maxv = pmax ? *(T*)pmax : std::numeric_limits<T>::max();
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if (px[i] < minv)
				py[i] = minv;
			else if (px[i] > maxv)
				py[i] = maxv;
			else
				py[i] = px[i];
		}
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 12) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (n->opset >= 11) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 6) {
		}else if (n->opset >= 1) {
		}
	}
};

} // namespace {

void resolver_default_op_Clip(node_t* n)
{
	n->ope = std::make_shared<Clip_operator>();
}

} // namespace onnx
