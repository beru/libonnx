#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Multinomial_operator : public operator_t {
	tensor_type_t dtype;
	int sample_size;
	float seed;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		dtype = (tensor_type_t)attribute("dtype", ONNX_TENSOR_TYPE_INT32);
		sample_size = attribute("sample_size", 1);
		seed = attribute("seed", 0.0f);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x, dtype);
	}

	template <typename XT, typename YT>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const int bsz = x->dims[0];
		const int csz = x->dims[1];
		const XT* px = (const XT*)x->data;
		std::vector<XT> cum(csz);

		if (seed != 0.0) {
			srand((unsigned int)seed);
		}

		YT* py = (YT*)y->data;
		for (int i = 0; i < bsz; ++i) {
			for (int j = 0; j < sample_size; ++j) {
				cum[0] = px[i * csz];
				for (int k = 1; k < csz; ++k) {
					cum[k] = cum[k - 1] + px[i * csz + k];
				}
				int l = csz - 1;
				for (int k = 0; k < csz - 1; ++k) {
					if ((XT)rand() / (XT)(RAND_MAX) < cum[k]) {
						l = k;
						break;
					}
				}
				int o = i * csz + l;
				py[o]++;
			}
		}
	}

	template <typename XT>
	void exec() {
		tensor_t* y = outputs[0];
		switch (y->type) {
		case ONNX_TENSOR_TYPE_INT32:
			exec<XT, int32_t>();
			break;
		case ONNX_TENSOR_TYPE_INT64:
			exec<XT, int64_t>();
			break;
		default:
			break;
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 7) {
			typed_exec<Multinomial_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Multinomial(int opset)
{
	return new (std::nothrow) Multinomial_operator;
}

} // namespace onnx
