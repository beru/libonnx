#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Abs_operator : public operator_t {
	bool init() override {
		return is_inout_size(1, 1);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			if constexpr (std::is_signed_v<T>) {
				py[i] = abs(px[i]);
			}else {
				py[i] = px[i];
			}
		}
	}

	void exec() override {
		tensor_type_t type = n->inputs[0]->type;
#if 0
		if (n->opset >= 13) {
			switch (type) {
			case ONNX_TENSOR_TYPE_INT8: exec<int8_t>(); return;
			case ONNX_TENSOR_TYPE_INT16: exec<int16_t>(); return;
			case ONNX_TENSOR_TYPE_INT32: exec<int32_t>(); return;
			case ONNX_TENSOR_TYPE_INT64: exec<int64_t>(); return;
			case ONNX_TENSOR_TYPE_UINT8: exec<uint8_t>(); return;
			case ONNX_TENSOR_TYPE_UINT16: exec<uint16_t>(); return;
			case ONNX_TENSOR_TYPE_UINT32: exec<uint32_t>(); return;
			case ONNX_TENSOR_TYPE_UINT64: exec<uint64_t>(); return;
			case ONNX_TENSOR_TYPE_FLOAT16: exec<float16_t>(); return;
			case ONNX_TENSOR_TYPE_FLOAT32: exec<float>(); return;
			case ONNX_TENSOR_TYPE_FLOAT64: exec<double>(); return;
			case ONNX_TENSOR_TYPE_BFLOAT16: exec<bfloat16_t>(); return;
			}
		}else if (n->opset >= 6) {
			switch (type) {
			case ONNX_TENSOR_TYPE_INT8: exec<int8_t>(); return;
			case ONNX_TENSOR_TYPE_INT16: exec<int16_t>(); return;
			case ONNX_TENSOR_TYPE_INT32: exec<int32_t>(); return;
			case ONNX_TENSOR_TYPE_INT64: exec<int64_t>(); return;
			case ONNX_TENSOR_TYPE_UINT8: exec<uint8_t>(); return;
			case ONNX_TENSOR_TYPE_UINT16: exec<uint16_t>(); return;
			case ONNX_TENSOR_TYPE_UINT32: exec<uint32_t>(); return;
			case ONNX_TENSOR_TYPE_UINT64: exec<uint64_t>(); return;
			case ONNX_TENSOR_TYPE_FLOAT16: exec<float16_t>(); return;
			case ONNX_TENSOR_TYPE_FLOAT32: exec<float>(); return;
			case ONNX_TENSOR_TYPE_FLOAT64: exec<double>(); return;
			}
		}else if (n->opset >= 1) {
			switch (type) {
			case ONNX_TENSOR_TYPE_FLOAT16: exec<float16_t>(); return;
			case ONNX_TENSOR_TYPE_FLOAT32: exec<float>(); return;
			case ONNX_TENSOR_TYPE_FLOAT64: exec<double>(); return;
			}
		}
#else
		if (n->opset >= 13) {
			TYPED_EXEC(type,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t)
		}else if (n->opset >= 6) {
			TYPED_EXEC(type,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double)
		}else if (n->opset >= 1) {
			TYPED_EXEC(type,
				float16_t, float, double)
		}
#endif
	}
};

} // namespace {

void resolver_default_op_Abs(node_t* n)
{
	n->ope = new Abs_operator;
}

} // namespace onnx
