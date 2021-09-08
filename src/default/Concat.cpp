#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ope_pdata_t : public node_t::ope_pdata_t {
	int axis;
	int caxis;
};

bool Concat_init(node_t* n)
{
	if (!(n->inputs.size() >= 1 && n->outputs.size() == 1)) {
		return false;
	}
	auto pdat = std::make_shared<ope_pdata_t>();
	if (!pdat)
		return false;
	pdat->axis = n->attribute("axis", 1);
	n->priv = pdat;
	return true;
}

int Concat_reshape(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	tensor_t* x = n->inputs[0];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);

	pdat->caxis = pdat->axis;
	if (pdat->caxis < 0)
		pdat->caxis += ndim;
	if (pdat->caxis < 0 || pdat->caxis >= ndim)
		return 0;
	int s = x->dims[pdat->caxis];
	for (size_t i = 1; i < n->inputs.size(); i++) {
		int* pdims = &n->inputs[i]->dims[0];
		for (int j = 0; j < ndim; j++) {
			if (j == pdat->caxis)
				s += pdims[j];
			else if (x->dims[j] != pdims[j])
				return 0;
			dims[j] = pdims[j];
		}
	}
	dims[pdat->caxis] = s;
	return y->reshape(&dims[0], ndim, x->type);
}

void Concat_ope(node_t* n)
{
	auto pdat = std::static_pointer_cast<ope_pdata_t>(n->priv);
	tensor_t* y = n->outputs[0];
	tensor_t* x;
	int ybase;
	int ypitch;
	int xpitch;
	int i, j, k;
	int idx;
	size_t o, l;

	if (n->inputs[0]->type == ONNX_TENSOR_TYPE_STRING) {
		std::string* py = (std::string*)y->data;
		for (i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
			ypitch *= y->dims[i];
		for (idx = 0, ybase = 0; idx < n->inputs.size(); idx++) {
			x = n->inputs[idx];
			std::string* px = (std::string*)x->data;
			for (i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
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
		char* px;
		int sz = tensor_type_sizeof(n->inputs[0]);
		for (i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
			ypitch *= y->dims[i];
		for (idx = 0, ybase = 0; idx < n->inputs.size(); idx++)	{
			x = n->inputs[idx];
			px = (char*)x->data;
			for (i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
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

} // namespace

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
			n->ope = Concat_ope;
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
			n->ope = Concat_ope;
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
			n->ope = Concat_ope;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
		case ONNX_TENSOR_TYPE_FLOAT32:
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Concat_ope;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Concat_init;
		n->reshape = Concat_reshape;
	}

}

} // namespace onnx
