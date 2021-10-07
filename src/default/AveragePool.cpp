#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct AveragePool_operator : public operator_t {

	enum auto_pad_t {
		AUTO_PAD_NOTSET		= 0,
		AUTO_PAD_SAME_UPPER	= 1,
		AUTO_PAD_SAME_LOWER	= 2,
		AUTO_PAD_VALID		= 3,
	};

	auto_pad_t auto_pad;
	int ceil_mode = 0;
	int count_include_pad = 0;
	std::vector<int> kernels;
	std::vector<int> pads;
	std::vector<int> strides;

	int cpads[32] = {0};

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		int64_t* ints;
		switch (c_hash(attribute("auto_pad", "NOTSET"))) {
		case C_HASH(NOTSET):
			auto_pad = AUTO_PAD_NOTSET;
			break;
		case C_HASH(SAME_UPPER):
			auto_pad = AUTO_PAD_SAME_UPPER;
			break;
		case C_HASH(SAME_LOWER):
			auto_pad = AUTO_PAD_SAME_LOWER;
			break;
		case C_HASH(VALID):
			auto_pad = AUTO_PAD_VALID;
			break;
		default:
			auto_pad = AUTO_PAD_NOTSET;
			break;
		}
		ceil_mode = attribute("ceil_mode", 0);
		count_include_pad = attribute("count_include_pad", 0);
		int kernel_shape = attribute("kernel_shape", ints);
		if (kernel_shape < 0) {
			return false;
		}
		kernels.resize(kernel_shape);
		for (int i = 0; i < kernels.size(); i++) {
			kernels[i] = ints[i];
		}
		pads.resize(kernels.size() * 2);
		if (pads.size()) {
			int l = attribute("pads", ints);
			int i;
			for (i = 0; i < l; i++) {
				pads[i] = ints[i];
			}
			for (; i < pads.size(); i++) {
				pads[i] = 0;
			}
		}
		strides.resize(kernels.size());
		if (strides.size()) {
			int l = attribute("strides", ints);
			int i;
			for (i = 0; i < l; i++) {
				strides[i] = ints[i];
			}
			for (; i < strides.size(); i++) {
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
		case AUTO_PAD_NOTSET:
			memcpy(cpads, &pads[0], sizeof(int) * pads.size());
			break;
		case AUTO_PAD_SAME_UPPER:
			for (int i = 0; i < pads.size() / 2; i++) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + kernels[i] - x->dims[i + 2];
				cpads[i] = pad / 2;
				cpads[i + kernels.size()] = pad - cpads[i];
			}
			break;
		case AUTO_PAD_SAME_LOWER:
			for (int i = 0; i < pads.size() / 2; i++) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + kernels[i] - x->dims[i + 2];
				cpads[i + kernels.size()] = pad / 2;
				cpads[i] = pad - cpads[i + kernels.size()];
			}
			break;
		case AUTO_PAD_VALID:
			memset(cpads, 0, sizeof(int) * pads.size());
			break;
		default:
			break;
		}
		dims[0] = x->dims[0];
		dims[1] = x->dims[1];
		for (int i = 0; i < ndim - 2; i++) {
			switch (auto_pad) {
			case AUTO_PAD_NOTSET:
				if (ceil_mode) {
					dims[i + 2] = ceilf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - kernels[i]) / (float)strides[i] + 1);
				}else {
					dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - kernels[i]) / (float)strides[i] + 1);
				}
				break;
			case AUTO_PAD_SAME_UPPER:
			case AUTO_PAD_SAME_LOWER:
				dims[i + 2] = ceilf(x->dims[i + 2] / (float)strides[i]);
				break;
			case AUTO_PAD_VALID:
				dims[i + 2] = ceilf((x->dims[i + 2] - kernels[i] + 1) / (float)strides[i]);
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
		std::vector<int> k_dim(x->ndim - 2);
		std::vector<int> i_dim(x->ndim);
		std::vector<int> o_dim(x->ndim);
		std::vector<int> b_dim(x->ndim);
		int size = multiply_accumulate(&kernels[0], &kernels[x->ndim - 2], 1);
		do {
			for (int i = 2; i < x->ndim; i++) {
				b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
			}
			T sum = 0;
			int padcnt = 0;
			std::fill(k_dim.begin(), k_dim.end(), 0);
			do {
				i_dim[0] = o_dim[0];
				i_dim[1] = o_dim[1];
				for (int i = 2; i < x->ndim; ++i) {
					i_dim[i] = b_dim[i] + k_dim[i - 2];
				}
				bool ispad = false;
				int i;
				for (i = 0; i < x->ndim; i++) {
					if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
						ispad = true;
						break;
					}
				}
				if (i >= x->ndim) {
					sum += px[dim_offset(x->ndim, &i_dim[0], &x->dims[0])];
				}
				if (ispad) {
					padcnt++;
				}
			} while (dim_next(x->ndim - 2, &k_dim[0], &kernels[0]));
			if (count_include_pad) {
				sum /= size;
			}else {
				sum /= (size - padcnt);
			}
			py[dim_offset(x->ndim, &o_dim[0], &y->dims[0])] = sum;
		} while (dim_next(x->ndim, &o_dim[0], &y->dims[0]));
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 11) {
			typed_exec<AveragePool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 10) {
			typed_exec<AveragePool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 7) {
			typed_exec<AveragePool_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<AveragePool_operator,
				float16_t, float, double
			>(this, type);
		}
	}
};

} // namespace {

operator_t* resolver_default_op_AveragePool(int opset)
{
	return new AveragePool_operator;
}

} // namespace onnx
