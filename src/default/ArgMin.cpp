#include "onnx.h"
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
		axis = attribute("axis", 0);
		keepdims = attribute("keepdims", 1);
		select_last_index = attribute("select_last_index", 0);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		if (axis < 0) {
			axis += x->ndim;
		}
		if (axis < 0 || axis >= x->ndim) {
			return false;
		}
		dim = x->dims[axis];
		stride = x->strides[axis];
		if (keepdims) {
			dims = x->dims;
			dims[axis] = 1;
		}else {
			ndim = 0;
			for (int i = 0; i < x->ndim; ++i) {
				if (i != axis) {
					dims[ndim++]= x->dims[i];
				}
			}
		}
		return y->reshape(&dims[0], ndim, ONNX_TENSOR_TYPE_INT64);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		int64_t* py = (int64_t*)y->data;
		size_t len = x->ndata;
		size_t idx = 0;
		int cnt = 0;

		while (idx < len) {
			if (cnt < stride) {
				T minv;
				int64_t mini;
				int i;
				const T *p;
				for (minv = px[idx], mini = 0, i = 1, p = px + idx + stride; i < dim; i++, p += stride) {
					if (select_last_index) {
						if (*p <= minv) {
							minv = *p;
							mini = i;
						}
					}else {
						if (*p < minv) {
							minv = *p;
							mini = i;
						}
					}
				}
				*py++ = mini;
				idx++;
				cnt++;
			}else {
				idx += (dim - 1) * stride;
				cnt = 0;
			}
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<ArgMax_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			>(this, type);
		}else if (opset >= 12) {
			typed_exec<ArgMax_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 11) {
			typed_exec<ArgMax_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<ArgMax_operator,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_ArgMin(int opset)
{
	return new (std::nothrow) ArgMax_operator;
}

} // namespace onnx
