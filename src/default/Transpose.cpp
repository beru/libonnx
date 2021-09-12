#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Transpose_operator : public operator_t {
	std::vector<int> perm;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		perm.resize(n->inputs[0]->ndim);
		int64_t* ints;
		if (perm.size() == n->attribute("perm", &ints)) {
			for (int i = 0; i < perm.size(); i++)
				perm[i] = ints[i];
		}else {
			for (int i = 0; i < perm.size(); i++)
				perm[i] = perm.size() - i - 1;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		if (!y->reshape_identity(x)) {
			return false;
		}
		for (int i = 0; i < x->ndim; i++)
			y->dims[i] = x->dims[perm[i]];
		return true;
	}

	template <typename T>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		size_t nperm = perm.size();
		std::vector<int> ix(nperm), iy(nperm);
		int ox, oy;
		size_t l;

		for (oy = 0, l = y->ndata; oy < l; oy++) {
			y->offset_to_indices(oy, &iy[0]);
			for (size_t i = 0; i < nperm; i++)
				ix[perm[i]] = iy[i];
			ox = x->indices_to_offset(&ix[0]);
			py[oy] = px[ox];
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
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			)
		}
	}

};

} // namespace {

void resolver_default_op_Transpose(node_t* n)
{
	n->ope = std::make_shared<Transpose_operator>();
}

} // namespace onnx
