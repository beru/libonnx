#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Multinomial_operator : public operator_t {
	tensor_type_t dtype;
	int sample_size;
	float seed;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		dtype = (tensor_type_t)n->attribute("dtype", ONNX_TENSOR_TYPE_INT32);
		sample_size = n->attribute("sample_size", 1);
		seed = n->attribute("seed", 0.0f);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		return y->reshape_identity(x, dtype);
	}

	template <typename YT>
	void exec() {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int bsz = x->dims[0];
		int csz = x->dims[1];
		const T* px = (const T*)x->data;
		std::vector<T> cum(csz);

		if (seed != 0.0)
			srand(seed);

		YT* py = (YT*)y->data;
		for (int i = 0; i < bsz; i++) {
			for (int j = 0; j < sample_size; j++) {
				cum[0] = px[i * csz];
				for (int k = 1; k < csz; k++)
					cum[k] = cum[k - 1] + px[i * csz + k];
				int l = csz - 1;
				for (int k = 0; k < csz - 1; k++) {
					if ((T)rand() / (T)(RAND_MAX) < cum[k]) {
						l = k;
						break;
					}
				}
				int o = i * csz + l;
				py[o]++;
			}
		}
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		switch (y->type) {
		case ONNX_TENSOR_TYPE_INT32:
			exec<int32_t>();
			break;
		case ONNX_TENSOR_TYPE_INT64:
			exec<int64_t>();
			break;
		default:
			break;
		}
	}
};

void resolver_default_op_Multinomial(node_t* n)
{
	if (n->opset >= 7) {
		n->ope = ope_type_select<Multinomial_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
