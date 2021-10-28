#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Squeeze_1_11_operator : public operator_t {

	bool init() override {
		return is_inout_size(1, 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		std::vector<int> dims(x->ndim);
		int ndim = 0;

		int64_t* axes = nullptr;
		int naxes = attribute("axes", axes);
		for (int i = 0; i < x->ndim; ++i) {
			if (x->dims[i] > 1) {
				dims[ndim++] = x->dims[i];
			}else {
				bool flag = false;
				for (int j = 0; j < naxes; ++j) {
					int axis = axes[j];
					if (axis < 0) {
						axis += x->ndim;
					}
					if (i == axis) {
						flag = true;
						break;
					}
				}
				if (!flag) {
					dims[ndim++] = x->dims[i];
				}
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	bool exec_impl() {
		copy_data(outputs[0], inputs[0]);
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
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
			return false;
			break;
		}
	}

};

struct Squeeze_13_operator : public operator_t {

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
			for (int i = 0; i < x->ndim; ++i) {
				if (x->dims[i] > 1) {
					dims[ndim++] = x->dims[i];
				}else {
					bool flag = false;
					for (size_t j = 0; j < a->ndata; ++j) {
						int axis = pa[j];
						if (axis < 0) {
							axis += x->ndim;
						}
						if (i == axis) {
							flag = true;
							break;
						}
					}
					if (!flag) {
						dims[ndim++] = x->dims[i];
					}
				}
			}
		}else {
			for (int i = 0; i < x->ndim; ++i) {
				if (x->dims[i] > 1) {
					dims[ndim++] = x->dims[i];
				}
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	bool exec_impl() {
		copy_data(outputs[0], inputs[0]);
		return true;
	}

	bool exec() override {
		tensor_type_t type = inputs[0]->type;
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
			return false;
			break;
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Squeeze(int opset)
{
	if (opset >= 13) {
		return new (std::nothrow) Squeeze_13_operator;
	}else {
		return new (std::nothrow) Squeeze_1_11_operator;
	}
}

} // namespace onnx
