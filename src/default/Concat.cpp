#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Concat_operator : public operator_t {
	int axis;
	int caxis;

	bool init() override {
		if (!(inputs.size() >= 1 && outputs.size() == 1)) {
			return false;
		}
		axis = attribute("axis", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const int ndim = x->ndim;
		std::vector<int> dims(ndim);

		caxis = axis;
		if (caxis < 0) {
			caxis += ndim;
		}
		if (caxis < 0 || caxis >= ndim) {
			return false;
		}
		int s = x->dims[caxis];
		for (size_t i = 1; i < inputs.size(); ++i) {
			const int* pdims = &inputs[i]->dims[0];
			for (int j = 0; j < ndim; ++j) {
				if (j == caxis) {
					s += pdims[j];
				}else if (x->dims[j] != pdims[j]) {
					return false;
				}
				dims[j] = pdims[j];
			}
		}
		dims[caxis] = s;
		return y->reshape(&dims[0], ndim, x->type);
	}
	
	void exec_impl() {
		tensor_t* y = outputs[0];

		if (inputs[0]->type == ONNX_TENSOR_TYPE_STRING) {
			std::string* py = (std::string*)y->data;
			int ypitch = multiply_accumulate(&y->dims[caxis], &y->dims[y->ndim], 1);
			int ybase = 0;
			for (size_t idx = 0; idx < inputs.size(); ++idx) {
				const tensor_t* x = inputs[idx];
				const std::string* px = (const std::string*)x->data;
				int xpitch = multiply_accumulate(&x->dims[caxis], &x->dims[x->ndim], 1);
				for (int o = 0, j = 0, k = ybase, l = x->ndata; o < l; ++o) {
					py[k + o] = px[o];
					if (++j == xpitch) {
						k += (ypitch - xpitch);
						j = 0;
					}
				}
				ybase += xpitch;
			}
		}else {
			char* py = (char*)y->data;
			const size_t sz = tensor_type_sizeof(inputs[0]);
			int ypitch = multiply_accumulate(&y->dims[caxis], &y->dims[y->ndim], 1);
			int ybase = 0;
			for (size_t idx = 0; idx < inputs.size(); ++idx) {
				const tensor_t* x = inputs[idx];
				const char* px = (const char*)x->data;
				int xpitch = multiply_accumulate(&x->dims[caxis], &x->dims[x->ndim], 1);
				size_t l = x->ndata;
				for (int o = 0, j = 0, k = ybase; o < l; ++o) {
					memcpy(py + (k + o) * sz, px + o * sz, sz);
					if (++j == xpitch) {
						k += (ypitch - xpitch);
						j = 0;
					}
				}
				ybase += xpitch;
			}
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
		}else if (opset >= 4) {
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
			switch (type) {
			case ONNX_TENSOR_TYPE_FLOAT16:
			case ONNX_TENSOR_TYPE_FLOAT32:
			case ONNX_TENSOR_TYPE_FLOAT64:
				exec_impl();
				break;
			default:
				break;
			}
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Concat(int opset)
{
	return new Concat_operator;
}

} // namespace onnx
