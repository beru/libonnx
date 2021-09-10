#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
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
		if (!is_inout_size(n, 1, 1)) {
			return false;
		}
		int i, l;
		int64_t* ints;
		switch (C_HASH(n->attribute("auto_pad", "NOTSET"))) {
		case C_HASH("NOTSET"):
			auto_pad = AUTO_PAD_NOTSET;
			break;
		case C_HASH("SAME_UPPER"):
			auto_pad = AUTO_PAD_SAME_UPPER;
			break;
		case C_HASH("SAME_LOWER"):
			auto_pad = AUTO_PAD_SAME_LOWER;
			break;
		case C_HASH("VALID"):
			auto_pad = AUTO_PAD_VALID;
			break;
		default:
			auto_pad = AUTO_PAD_NOTSET;
			break;
		}
		ceil_mode = n->attribute("ceil_mode", 0);
		count_include_pad = n->attribute("count_include_pad", 0);
		int kernel_shape = n->attribute("kernel_shape", &ints);
		if (kernel_shape < 0)
			return false;
		kernels.resize(kernel_shape);
		for (i = 0; i < kernels.size(); i++)
			kernels[i] = ints[i];
		pads.resize(kernels.size() * 2);
		if (pads.size()) {
			l = n->attribute("pads", &ints);
			for (i = 0; i < l; i++)
				pads[i] = ints[i];
			for (; i < pads.size(); i++)
				pads[i] = 0;
		}
		strides.resize(kernels.size());
		if (strides.size()) {
			l = n->attribute("strides", &ints);
			for (i = 0; i < l; i++)
				strides[i] = ints[i];
			for (; i < strides.size(); i++)
				strides[i] = 1;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		int ndim = x->ndim;
		std::vector<int> dims(ndim);
		int pad;
		int i;

		switch (auto_pad) {
		case AUTO_PAD_NOTSET:
			memcpy(cpads, &pads[0], sizeof(int) * pads.size());
			break;
		case AUTO_PAD_SAME_UPPER:
			for (i = 0; i < pads.size() / 2; i++) {
				pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + kernels[i] - x->dims[i + 2];
				cpads[i] = pad / 2;
				cpads[i + kernels.size()] = pad - cpads[i];
			}
			break;
		case AUTO_PAD_SAME_LOWER:
			for (i = 0; i < pads.size() / 2; i++) {
				pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + kernels[i] - x->dims[i + 2];
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
		for (i = 0; i < ndim - 2; i++) {
			switch (auto_pad) {
			case AUTO_PAD_NOTSET:
				if (ceil_mode)
					dims[i + 2] = ceilf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - kernels[i]) / (float)strides[i] + 1);
				else
					dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - kernels[i]) / (float)strides[i] + 1);
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

	void exec() override {
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T sum;
		std::vector<int> k_dim(x->ndim - 2);
		std::vector<int> i_dim(x->ndim);
		std::vector<int> o_dim(x->ndim);
		std::vector<int> b_dim(x->ndim);
		int padcnt, ispad, size;
		int i;

		for (i = 0, size = 1; i < x->ndim - 2; ++i) {
			size *= kernels[i];
		}
		do {
			for (i = 2; i < x->ndim; i++)
				b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
			sum = 0;
			padcnt = 0;
			std::fill(k_dim.begin(), k_dim.end(), 0);
			do {
				i_dim[0] = o_dim[0];
				i_dim[1] = o_dim[1];
				for (i = 2; i < x->ndim; ++i)
					i_dim[i] = b_dim[i] + k_dim[i - 2];
				ispad = 0;
				for (i = 0; i < x->ndim; i++) {
					if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
						ispad = 1;
						break;
					}
				}
				if (i >= x->ndim)
					sum += px[dim_offset(x->ndim, &i_dim[0], &x->dims[0])];
				if (ispad)
					padcnt++;
			} while (dim_next(x->ndim - 2, &k_dim[0], &kernels[0]));
			if (count_include_pad)
				sum /= size;
			else
				sum /= (size - padcnt);
			py[dim_offset(x->ndim, &o_dim[0], &y->dims[0])] = sum;
		} while (dim_next(x->ndim, &o_dim[0], &y->dims[0]));
	}
};

void resolver_default_op_AveragePool(node_t* n)
{
	if (n->opset >= 11) {
		n->ope = ope_type_select<AveragePool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = ope_type_select<AveragePool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<AveragePool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<AveragePool_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
