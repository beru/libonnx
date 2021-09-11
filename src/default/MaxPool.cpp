#include <onnx.h>
#include "util.h"

namespace onnx {

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
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
		if (!(n->inputs.size() == 1 && n->outputs.size() >= 1)) {
			return false;
		}
		int64_t* ints;
		int i, l;
		switch (C_HASH(n->attribute("auto_pad", "NOTSET")))	{
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
		storage_order = n->attribute("storage_order", 0);
		nkernel = n->attribute("kernel_shape", &ints);
		if (nkernel > 0) {
			kernels.resize(nkernel);
			for (i = 0; i < nkernel; i++)
				kernels[i] = ints[i];
		}
		ndilation = nkernel;
		dilations.resize(ndilation);
		if (ndilation > 0) {
			l = n->attribute("dilations", &ints);
			for (i = 0; i < l; i++)
				dilations[i] = ints[i];
			for (; i < ndilation; i++)
				dilations[i] = 1;
		}
		npad = nkernel * 2;
		pads.resize(npad);
		if (npad > 0) {
			l = n->attribute("pads", &ints);
			for (i = 0; i < l; i++)
				pads[i] = ints[i];
			for (; i < npad; i++)
				pads[i] = 0;
		}
		nstride = nkernel;
		strides.resize(nstride);
		if (nstride > 0) {
			l = n->attribute("strides", &ints);
			for (i = 0; i < l; i++)
				strides[i] = ints[i];
			for (; i < nstride; i++)
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
			memcpy(cpads, &pads[0], sizeof(int) * npad);
			break;
		case AUTO_PAD_SAME_UPPER:
			for (i = 0; i < npad / 2; i++) {
				pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
				cpads[i] = pad / 2;
				cpads[i + nkernel] = pad - cpads[i];
			}
			break;
		case AUTO_PAD_SAME_LOWER:
			for (i = 0; i < npad / 2; i++) {
				pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
				cpads[i + nkernel] = pad / 2;
				cpads[i] = pad - cpads[i + nkernel];
			}
			break;
		case AUTO_PAD_VALID:
			memset(cpads, 0, sizeof(int) * npad);
			break;
		default:
			break;
		}
		dims[0] = x->dims[0];
		dims[1] = x->dims[1];
		for (i = 0; i < ndim - 2; i++) {
			switch (auto_pad)	{
			case AUTO_PAD_NOTSET:
				if (ceil_mode)
					dims[i + 2] = ceilf((x->dims[i + 2] + cpads[i] + cpads[i + nkernel] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
				else
					dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + nkernel] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
				break;
			case AUTO_PAD_SAME_UPPER:
			case AUTO_PAD_SAME_LOWER:
				dims[i + 2] = ceilf(x->dims[i + 2] / (float)strides[i]);
				break;
			case AUTO_PAD_VALID:
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
		const tensor_t* x = n->inputs[0];
		tensor_t* y = n->outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;
		T maxv, v;
		std::vector<int> k_dim(x->ndim - 2);
		std::vector<int> i_dim(x->ndim);
		std::vector<int> o_dim(x->ndim);
		std::vector<int> b_dim(x->ndim);
		int i;

		do {
			for (i = 2; i < x->ndim; ++i)
				b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
			maxv = std::numeric_limits<T>::min();
			std::fill(k_dim.begin(), k_dim.end(), 0);
			do {
				i_dim[0] = o_dim[0];
				i_dim[1] = o_dim[1];
				for (i = 2; i < x->ndim; ++i)
					i_dim[i] = b_dim[i] + k_dim[i - 2];
				for (i = 0; i < x->ndim; ++i) {
					if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i]))
						break;
				}
				if (i >= x->ndim) {
					v = px[dim_offset(x->ndim, &i_dim[0], &x->dims[0])];
					maxv = max(v, maxv);
				}
			} while (dim_next(x->ndim - 2, &k_dim[0], &kernels[0]));
			py[dim_offset(x->ndim, &o_dim[0], &y->dims[0])] = maxv;
		} while (dim_next(x->ndim, &o_dim[0], &y->dims[0]));
	}

	void exec() override {
		if (n->opset >= 12) {
			typed_exec<MaxPool_operator,
				int8_t,
				uint8_t,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 11) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 10) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 8) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
			typed_exec<MaxPool_operator,
				float16_t, float, double
			>(n->inputs[0]->type);
		}
	}

};

void resolver_default_op_MaxPool(node_t* n)
{
	n->ope = std::make_shared<MaxPool_operator>();
}

} // namespace onnx
