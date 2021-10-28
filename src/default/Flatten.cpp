#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Flatten_operator : public operator_t {
	int axis;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		axis = attribute("axis", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		std::vector<int> dims(x->ndim);

		if (axis < 0) {
			axis += x->ndim;
		}
		if (axis < 0 || axis >= x->ndim) {
			return false;
		}
		int j = 1;
		int ndim = 0;
		for (int i = 0; i < x->ndim; ++i) {
			if (i != axis) {
				j *= x->dims[i];
			}else {
				dims[ndim++] = j;
				j = x->dims[i];
			}
		}
		dims[ndim++] = j;
		return y->reshape(&dims[0], ndim, x->type);
	}

	bool exec_impl() {
		copy_data(outputs[0], inputs[0]);
		return true;
	}

	bool exec() override {
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
				return exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 11) {
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
			case ONNX_TENSOR_TYPE_FLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT32:
			case ONNX_TENSOR_TYPE_FLOAT64:
			case ONNX_TENSOR_TYPE_COMPLEX64:
			case ONNX_TENSOR_TYPE_COMPLEX128:
			case ONNX_TENSOR_TYPE_STRING:
				return exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 9) {
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
			case ONNX_TENSOR_TYPE_FLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT32:
			case ONNX_TENSOR_TYPE_FLOAT64:
			case ONNX_TENSOR_TYPE_COMPLEX64:
			case ONNX_TENSOR_TYPE_COMPLEX128:
			case ONNX_TENSOR_TYPE_STRING:
				return exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 1) {
			switch (type) {
			case ONNX_TENSOR_TYPE_FLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT32:
			case ONNX_TENSOR_TYPE_FLOAT64:
				return exec_impl();
				break;
			default:
				break;
			}
		}
		return false;
	}
};

} // namespace {

operator_t* resolver_default_op_Flatten(int opset) { return new (std::nothrow) Flatten_operator; }

} // namespace onnx
