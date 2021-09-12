#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct ArgMax_operator : public operator_t {
	int axis;
	int keepdims;
	int select_last_index;

	int dim;
	int stride;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		axis = n->attribute("axis", 0);
		keepdims = n->attribute("keepdims", 1);
		select_last_index = n->attribute("select_last_index", 0);
		return true;
	}
	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		if (axis < 0)
			axis += x->ndim;
		if (axis < 0 || axis >= x->ndim)
			return false;
		dim = x->dims[axis];
		stride = x->strides[axis];
		if (keepdims)	{
			dims = x->dims;
			dims[axis] = 1;
		}else {
			for (int i = 0, ndim = 0; i < x->ndim; i++)	{
				if (i != axis)
					dims[ndim++]= x->dims[i];
			}
		}
		return y->reshape(&dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T *p;
		const T* px = (const T*)x->data;
		T maxv;
		int64_t* py = (int64_t*)y->data;
		int64_t maxi;
		size_t len = x->ndata;
		size_t idx = 0;
		int cnt = 0;
		int i;

		while (idx < len) {
			if (cnt < stride)	{
				for (maxv = px[idx], maxi = 0, i = 1, p = px + idx + stride; i < dim; i++, p += stride) {
					if (select_last_index) {
						if (*p >= maxv) {
							maxv = *p;
							maxi = i;
						}
					}else {
						if (*p > maxv) {
							maxv = *p;
							maxi = i;
						}
					}
				}
				*py++ = maxi;
				idx++;
				cnt++;
			}else {
				idx += (dim - 1) * stride;
				cnt = 0;
			}
		}
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (n->opset >= 12) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (n->opset >= 11) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			)
		}
	}
};

} // namespace {

void resolver_default_op_ArgMax(node_t* n)
{
	n->ope = std::make_shared<ArgMax_operator>();
}

} // namespace onnx
