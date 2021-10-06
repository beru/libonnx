#include "onnx.h"
#include "util.h"
	
namespace onnx {

namespace {

struct Unsqueeze_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* a = inputs[1];
		const int64_t* pa = (const int64_t*)a->data;
		int ndim = x->ndim + a->ndata;
		std::vector<int> dims(ndim);

		for (int i = 0; i < a->ndata; i++) {
			int axis = pa[i];
			if (axis < 0) {
				axis += ndim;
			}
			if (axis >= 0 && axis < ndim) {
				dims[axis] = 1;
			}
		}
		for (int i = 0, j = 0; i < ndim; i++) {
			if (dims[i] != 1) {
				dims[i] = x->dims[j++];
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

operator_t* resolver_default_op_Unsqueeze(int opset)
{
	return new Unsqueeze_operator;
}

} // namespace onnx
