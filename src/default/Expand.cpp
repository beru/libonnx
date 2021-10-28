#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Expand_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* s = inputs[1];
		const int64_t* ps = (const int64_t*)s->data;
		const int ndim = max(x->ndim, (int)s->ndata);
		std::vector<int> dims(ndim);

		for (int i = x->ndim - 1, j = s->ndata - 1, k = ndim - 1; k >= 0; k--) {
			if (i < 0) {
				dims[k] = ps[j--];
			}else if (j < 0) {
				dims[k] = x->dims[i--];
			}else {
				if (x->dims[i] == ps[j]) {
					dims[k] = x->dims[i];
				}else if ((x->dims[i] == 1) || (ps[j] == 1)) {
					dims[k] = (x->dims[i] > ps[j]) ? x->dims[i] : ps[j];
				}else {
					return false;
				}
				i--;
				j--;
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		T* py = (T*)y->data;
		const T* px = (const T*)x->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			px = (const T*)x->broadcast_map_address(y, i);
			py[i] = *px;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Expand_operator,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}else if (opset >= 8) {
			typed_exec<Expand_operator,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Expand(int opset) { return new (std::nothrow) Expand_operator; }

} // namespace onnx
