#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct MatMul_operator : public operator_t {
	int m = 0;
	int n = 0;
	int k = 0;

	bool init() override {
		if (!is_inout_size(2, 1)) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		tensor_t* y = operator_t::n->outputs[0];
		const tensor_t* a = operator_t::n->inputs[0];
		const tensor_t* b = operator_t::n->inputs[1];
		std::vector<int> adims;
		std::vector<int> bdims;

		if (a->ndim == 1) {
			adims.resize(2);
			adims[0] = 1;
			adims[1] = a->dims[0];
		}else {
			adims = a->dims;
		}
		if (b->ndim == 1) {
			bdims[0] = b->dims[0];
			bdims[1] = 1;
		}else {
			bdims = b->dims;
		}
		int ndim = max(adims.size(), bdims.size());
		std::vector<int> dims(ndim);
		if (adims.size() < 2 || bdims.size() < 2)
			return false;
		if (adims[adims.size() - 1] != bdims[bdims.size() - 2])
			return false;
		dims[ndim - 2] = adims[adims.size() - 2];
		dims[ndim - 1] = bdims[bdims.size() - 1];
		for (int i = 3; i <= ndim; i++) {
			int alen = (adims.size() - i) < 0 ? 1 : adims[adims.size() - i];
			int blen = (bdims.size() - i) < 0 ? 1 : bdims[bdims.size() - i];
			if (alen != blen && alen > 1 && blen > 1)
				return false;
			dims[ndim - i] = max(alen, blen);
		}
		m = adims[adims.size() - 2];
		n = bdims[bdims.size() - 1];
		k = adims[adims.size() - 1];
		return y->reshape(&dims[0], ndim, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = operator_t::n->outputs[0];
		const tensor_t* a = operator_t::n->inputs[0];
		const tensor_t* b = operator_t::n->inputs[1];
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i += m * n) {
			T* pa = (T*)a->broadcast_map_address(y, i);
			T* pb = (T*)b->broadcast_map_address(y, i);
			for (int u = 0; u < m; u++) {
				for (int v = 0; v < n; v++) {
					T sum = 0;
					for (int w = 0; w < k; w++)
						sum += pa[u * k + w] * pb[w * n + v];
					py[i + u * n + v] = sum;
				}
			}
		}
	}

	void exec() override {
		auto opset = operator_t::n->opset;
		auto input_type = operator_t::n->inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(input_type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 9) {
			TYPED_EXEC(input_type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 1) {
			TYPED_EXEC(input_type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_MatMul(node_t* n)
{
	n->ope = std::make_shared<MatMul_operator>();
}

} // namespace onnx
