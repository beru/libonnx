#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Squeeze_operator : public operator_t {

	bool init() override {
		return (inputs.size() >= 1) && (outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		std::vector<int> dims(x->ndim);
		int ndim = 0;

		if (inputs.size() > 1) {
			const tensor_t* a = inputs[1];
			const int64_t* pa = (const int64_t*)a->data;
			for (int i = 0; i < x->ndim; i++) {
				if (x->dims[i] > 1)
					dims[ndim++] = x->dims[i];
				else {
					bool flag = false;
					for (int j = 0; j < a->ndata; j++) {
						int axis = pa[j];
						if (axis < 0)
							axis += x->ndim;
						if (i == axis) {
							flag = true;
							break;
						}
					}
					if (!flag)
						dims[ndim++] = x->dims[i];
				}
			}
		}else {
			for (int i = 0; i < x->ndim; i++) {
				if (x->dims[i] > 1)
					dims[ndim++] = x->dims[i];
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	void exec_impl() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		if (x->type == ONNX_TENSOR_TYPE_STRING) {
			const std::string* px = (const std::string*)x->data;
			std::string* py = (std::string*)y->data;
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				py[i] = px[i];
			}
		}else {
			memcpy(y->data, x->data, x->ndata * tensor_type_sizeof(x));
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			switch (type) {
			case ONNX_TENSOR_TYPE_BOOL:
			case ONNX_TENSOR_TYPE_INT8:
			case ONNX_TENSOR_TYPE_INT16:
			case ONNX_TENSOR_TYPE_INT32:
			case ONNX_TENSOR_TYPE_INT64:
			case ONNX_TENSOR_TYPE_UINT8:
			case ONNX_TENSOR_TYPE_UINT16:
			case ONNX_TENSOR_TYPE_UINT32:
			case ONNX_TENSOR_TYPE_UINT64:
			case ONNX_TENSOR_TYPE_BFLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT32:
			case ONNX_TENSOR_TYPE_FLOAT64:
			case ONNX_TENSOR_TYPE_COMPLEX64:
			case ONNX_TENSOR_TYPE_COMPLEX128:
			case ONNX_TENSOR_TYPE_STRING:
				exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 11) {
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Squeeze()
{
	return new Squeeze_operator;
}

} // namespace onnx
