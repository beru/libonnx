#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Reshape_operator : public operator_t {

	bool init() override {
		if (!is_inout_size(2, 1)) {
			return false;
		}
		const tensor_t* x = inputs[0];
		const tensor_t* s = inputs[1];
		if ((x->ndim == 0) || (x->type == ONNX_TENSOR_TYPE_UNDEFINED))
			return false;
		if ((s->ndim == 0) || (s->type != ONNX_TENSOR_TYPE_INT64))
			return false;
		return true;
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* s = inputs[1];
		int64_t* ps = (int64_t*)s->data;
		int total_dim = 1;
		int total_shape = 1;
		int ndim = s->ndata;
		std::vector<int> dims(ndim);

		for (int i = 0; i < ndim; i++) {
			if (ps[i] == 0)
				dims[i] = x->dims[i];
			else if (ps[i] > 0)
				dims[i] = ps[i];
			else {
				for (int j = 0; j < x->ndim; j++)
					total_dim *= x->dims[j];
				for (int j = 0; j < ndim; j++) {
					if (ps[j] > 0)
						total_shape *= ps[j];
					else if (ps[j] == 0)
						total_shape *= x->dims[j];
				}
				dims[i] = total_dim / total_shape;
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	void exec_impl() {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		if (x->type == ONNX_TENSOR_TYPE_STRING) {
			std::string* py = (std::string*)y->data;
			const std::string* px = (const std::string*)x->data;
			for (size_t i = 0, l = y->ndata; i < l; i++) {
				py[i] = px[i];
			}
		}else {
			memcpy(y->data, x->data, x->ndata * tensor_type_sizeof(x));
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 14) {
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
		}else if (opset >= 13) {
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
		}else if (opset >= 5) {
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
				exec_impl();
				break;
			default:
				break;
			}
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Reshape()
{
	return new Reshape_operator;
}

} // namespace onnx
