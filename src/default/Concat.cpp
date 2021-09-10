#include <onnx.h>
#include "util.h"

namespace onnx {

struct Concat_operator : public operator_t {
	int axis;
	int caxis;

	bool init() override {
		if (!(n->inputs.size() >= 1 && n->outputs.size() == 1)) {
			return false;
		}
		axis = n->attribute("axis", 1);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		caxis = axis;
		if (caxis < 0)
			caxis += ndim;
		if (caxis < 0 || caxis >= ndim)
			return false;
		int s = x->dims[caxis];
		for (size_t i = 1; i < n->inputs.size(); i++) {
			int* pdims = &n->inputs[i]->dims[0];
			for (int j = 0; j < ndim; j++) {
				if (j == caxis)
					s += pdims[j];
				else if (x->dims[j] != pdims[j])
					return false;
				dims[j] = pdims[j];
			}
		}
		dims[caxis] = s;
		return y->reshape(&dims[0], ndim, x->type);
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* x;
		int ybase;
		int ypitch;
		int xpitch;
		int i, j, k;
		int idx;
		size_t o, l;

		if (n->inputs[0]->type == ONNX_TENSOR_TYPE_STRING) {
			std::string* py = (std::string*)y->data;
			for (i = y->ndim - 1, ypitch = 1; i >= caxis; i--)
				ypitch *= y->dims[i];
			for (idx = 0, ybase = 0; idx < n->inputs.size(); idx++) {
				x = n->inputs[idx];
				const std::string* px = (const std::string*)x->data;
				for (i = x->ndim - 1, xpitch = 1; i >= caxis; i--)
					xpitch *= x->dims[i];
				for (o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++) {
					py[k + o] = px[o];
					if (++j == xpitch) 	{
						k += (ypitch - xpitch);
						j = 0;
					}
				}
				ybase += xpitch;
			}
		}else {
			char* py = (char*)y->data;
			const char* px;
			int sz = tensor_type_sizeof(n->inputs[0]);
			for (i = y->ndim - 1, ypitch = 1; i >= caxis; i--)
				ypitch *= y->dims[i];
			for (idx = 0, ybase = 0; idx < n->inputs.size(); idx++)	{
				x = n->inputs[idx];
				px = (const char*)x->data;
				for (i = x->ndim - 1, xpitch = 1; i >= caxis; i--)
					xpitch *= x->dims[i];
				for (o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++)	{
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
};

void resolver_default_op_Concat(node_t* n)
{
	if (n->opset >= 13) {
		switch (n->inputs[0]->type) {
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
			n->ope = std::make_shared<Concat_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 11) {
		switch (n->inputs[0]->type) {
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
			n->ope = std::make_shared<Concat_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 4) {
		switch (n->inputs[0]->type) {
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
			n->ope = std::make_shared<Concat_operator>();
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = std::make_shared<Concat_operator>();
			break;
		default:
			break;
		}
	}
}

} // namespace onnx
