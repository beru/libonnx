#include <onnx.h>
#include "util.h"

namespace onnx {

struct Squeeze_operator : public operator_t {

	bool init() override {
		return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* x = n->inputs[0];
		const tensor_t* a;
		const int64_t* pa;
		std::vector<int> dims(x->ndim);
		int ndim = 0;
		int axis, flag;
		int i, j;

		if (n->inputs.size() > 1) {
			a = n->inputs[1];
			pa = (const int64_t*)a->data;
			for (i = 0, ndim = 0; i < x->ndim; i++) {
				if (x->dims[i] > 1)
					dims[ndim++] = x->dims[i];
				else {
					for (j = 0, flag = 0; j < a->ndata; j++) {
						axis = pa[j];
						if (axis < 0)
							axis += x->ndim;
						if (i == axis) {
							flag = 1;
							break;
						}
					}
					if (!flag)
						dims[ndim++] = x->dims[i];
				}
			}
		}else {
			for (i = 0, ndim = 0; i < x->ndim; i++) {
				if (x->dims[i] > 1)
					dims[ndim++] = x->dims[i];
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
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

};

void resolver_default_op_Squeeze(node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type)	{
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
			n->ope = std::make_shared<Squeeze_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
