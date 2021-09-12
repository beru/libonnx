#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Tile_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* x = n->inputs[0];
		const tensor_t* r = n->inputs[1];
		const int64_t* pr = (const int64_t*)r->data;
		int ndim = x->ndim;
		std::vector<int> dims(ndim);

		for (int i = 0; i < ndim; i++)
			dims[i] = x->dims[i] * pr[i];
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* x = n->inputs[0];
		T* py = (T*)y->data;
		const T* px = (const T*)x->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			px = (const T*)x->broadcast_map_address(y, i);
			py[i] = *px;
		}
	}

	void exec() override {
		if (n->opset >= 13) {
			TYPED_EXEC(n->inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::complex<float>, std::complex<double>,
				std::string
			)
		}else if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			)
		}else if (n->opset >= 1) {
		}
	}
};

} // namespace {

void resolver_default_op_Tile(node_t* n)
{
	n->ope = std::make_shared<Tile_operator>();
}

} // namespace onnx
