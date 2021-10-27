#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

enum auto_pad_t {
	NOTSET,
	SAME_UPPER,
	SAME_LOWER,
	VALID,
};

struct MaxPool_operator : public operator_t {

	auto_pad_t auto_pad;
	int ceil_mode = 0;
	int storage_order = 0;
	std::vector<int> kernels;
	int nkernel = 0;
	std::vector<int> dilations;
	int ndilation = 0;
	std::vector<int> pads;
	int npad = 0;
	std::vector<int> strides;
	int nstride = 0;

	int cpads[32] = {0};

	bool init() override {
		if (!(inputs.size() == 1 && outputs.size() >= 1)) {
			return false;
		}
		int64_t* ints;
		int i, l;

		static const std::unordered_map<std::string_view, auto_pad_t> m0 {
			#define X(a) { #a, auto_pad_t:: a }
			X(NOTSET),
			X(SAME_UPPER),
			X(SAME_LOWER),
			X(VALID),
			#undef X
		};
		auto_pad = m0.at(attribute("auto_pad", "NOTSET"));
		ceil_mode = attribute("ceil_mode", 0);
		storage_order = attribute("storage_order", 0);
		nkernel = attribute("kernel_shape", ints);
		if (nkernel > 0) {
			kernels.resize(nkernel);
			for (i = 0; i < nkernel; ++i) {
				kernels[i] = ints[i];
			}
		}
		ndilation = nkernel;
		dilations.resize(ndilation);
		if (ndilation > 0) {
			l = attribute("dilations", ints);
			for (i = 0; i < l; ++i) {
				dilations[i] = ints[i];
			}
			for (; i < ndilation; ++i) {
				dilations[i] = 1;
			}
		}
		npad = nkernel * 2;
		pads.resize(npad);
		if (npad > 0) {
			l = attribute("pads", ints);
			for (i = 0; i < l; ++i) {
				pads[i] = ints[i];
			}
			for (; i < npad; ++i) {
				pads[i] = 0;
			}
		}
		nstride = nkernel;
		strides.resize(nstride);
		if (nstride > 0) {
			l = attribute("strides", ints);
			for (i = 0; i < l; ++i) {
				strides[i] = ints[i];
			}
			for (; i < nstride; ++i) {
				strides[i] = 1;
			}
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const int ndim = x->ndim;
		std::vector<int> dims(ndim);

		switch (auto_pad) {
		case NOTSET:
			memcpy(cpads, &pads[0], sizeof(int) * npad);
			break;
		case SAME_UPPER:
			for (int i = 0; i < npad / 2; ++i) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
				cpads[i] = pad / 2;
				cpads[i + nkernel] = pad - cpads[i];
			}
			break;
		case SAME_LOWER:
			for (int i = 0; i < npad / 2; ++i) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
				cpads[i + nkernel] = pad / 2;
				cpads[i] = pad - cpads[i + nkernel];
			}
			break;
		case VALID:
			memset(cpads, 0, sizeof(int) * npad);
			break;
		default:
			break;
		}
		dims[0] = x->dims[0];
		dims[1] = x->dims[1];
		for (int i = 0; i < ndim - 2; ++i) {
			switch (auto_pad) {
			case NOTSET:
				if (ceil_mode) {
					dims[i + 2] = ceilf((x->dims[i + 2] + cpads[i] + cpads[i + nkernel] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
				}else {
					dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + nkernel] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
				}
				break;
			case SAME_UPPER:
			case SAME_LOWER:
				dims[i + 2] = ceilf(x->dims[i + 2] / (float)strides[i]);
				break;
			case VALID:
				dims[i + 2] = ceilf((x->dims[i + 2] - ((kernels[i] - 1) * dilations[i] + 1) + 1) / (float)strides[i]);
				break;
			default:
				break;
			}
		}
		return y->reshape(&dims[0], ndim, x->type);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
#if 0
		std::vector<int> k_dim(x->ndim - 2);
		std::vector<int> i_dim(x->ndim);
		std::vector<int> o_dim(x->ndim);
		std::vector<int> b_dim(x->ndim);
#else
		assert(x->ndim < 8);
		int k_dim[8 - 2];
		int i_dim[8] = {0};
		int o_dim[8] = {0};
		int b_dim[8] = {0};
#endif
		int i;

		do {
			for (i = 2; i < x->ndim; ++i) {
				b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
			}
			T maxv = std::numeric_limits<T>::lowest();
//			std::fill(k_dim.begin(), k_dim.end(), 0);
			std::fill(&k_dim[0], &k_dim[6], 0);
			do {
				i_dim[0] = o_dim[0];
				i_dim[1] = o_dim[1];
				for (i = 2; i < x->ndim; ++i) {
					i_dim[i] = b_dim[i] + k_dim[i - 2];
				}
				for (i = 0; i < x->ndim; ++i) {
					if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
						break;
					}
				}
				if (i >= x->ndim) {
					T v = px[dim_offset(x->ndim, &i_dim[0], &x->dims[0])];
					maxv = max(v, maxv);
				}
			} while (dim_next(x->ndim - 2, &k_dim[0], &kernels[0]));
			py[dim_offset(x->ndim, &o_dim[0], &y->dims[0])] = maxv;
		} while (dim_next(x->ndim, &o_dim[0], &y->dims[0]));
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 12) {
			typed_exec<MaxPool_operator,
				int8_t,
				uint8_t,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 11) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 10) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 8) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_MaxPool(int opset)
{
	return new (std::nothrow) MaxPool_operator;
}

} // namespace onnx
