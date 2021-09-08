#include <onnx.h>
#include "refnd.h"
#include "util.h"

namespace {

enum auto_pad_t {
	AUTO_PAD_NOTSET		= 0,
	AUTO_PAD_SAME_UPPER	= 1,
	AUTO_PAD_SAME_LOWER	= 2,
	AUTO_PAD_VALID		= 3,
};

enum conv_mode_t {
	CONV_SIMPLE = 0,
	CONV_CACHED = 1,
	CONV_IM2COL = 2,
};

struct ope_pdata_t : public onnx_node_t::ope_pdata_t {
	auto_pad_t auto_pad = AUTO_PAD_NOTSET;
	int group = 0;
	std::vector<int> kernels;
	std::vector<int> dilations;
	std::vector<int> pads;
	std::vector<int> strides;

	int cpads[32] = {0};
};

bool Conv_init(onnx_node_t* n)
{
	if (!(n->inputs.size() >= 2 && n->outputs.size() == 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow)ope_pdata_t;
	if (!pdat)
		return false;
	int64_t* ints = nullptr;
	int i, l;
	switch (C_HASH(n->read_attribute("auto_pad", "NOTSET"))) {
	case C_HASH("NOTSET"):
		pdat->auto_pad = AUTO_PAD_NOTSET;
		break;
	case C_HASH("SAME_UPPER"):
		pdat->auto_pad = AUTO_PAD_SAME_UPPER;
		break;
	case C_HASH("SAME_LOWER"):
		pdat->auto_pad = AUTO_PAD_SAME_LOWER;
		break;
	case C_HASH("VALID"):
		pdat->auto_pad = AUTO_PAD_VALID;
		break;
	default:
		pdat->auto_pad = AUTO_PAD_NOTSET;
		break;
	}
	pdat->group = n->read_attribute("group", 1);
	int nkernel = n->read_attribute("kernel_shape", &ints);
	if (nkernel > 0) {
		pdat->kernels.resize(nkernel);
		for (i=0; i<nkernel; ++i) {
			pdat->kernels[i] = ints[i];
		}
		int ndilation = nkernel;
		pdat->dilations.resize(ndilation);
		l = n->read_attribute("dilations", &ints);
		for (i = 0; i < l; i++)
			pdat->dilations[i] = ints[i];
		for (; i < ndilation; i++)
			pdat->dilations[i] = 1;
		int npad = nkernel * 2;
		pdat->pads.resize(npad);
		l = n->read_attribute("pads", &ints);
		for (i = 0; i < l; i++)
			pdat->pads[i] = ints[i];
		for (; i < npad; i++)
			pdat->pads[i] = 0;
		int nstride = nkernel;
		pdat->strides.resize(nstride);
		l = n->read_attribute("strides", &ints);
		for (i = 0; i < l; i++)
			pdat->strides[i] = ints[i];
		for (; i < nstride; i++)
			pdat->strides[i] = 1;
	}
	n->priv = pdat;
	return true;
}

int Conv_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* w = n->inputs[1];
	int ndim = x->ndim;
	std::vector<int> dims(ndim);
	int pad;
	int i;

	switch (pdat->auto_pad) {
	case AUTO_PAD_NOTSET:
		memcpy(pdat->cpads, &pdat->pads[0], sizeof(int) * pdat->pads.size());
		break;
	case AUTO_PAD_SAME_UPPER:
		for (i = 0; i < pdat->pads.size() / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i] = pad / 2;
			pdat->cpads[i + pdat->kernels.size()] = pad - pdat->cpads[i];
		}
		break;
	case AUTO_PAD_SAME_LOWER:
		for (i = 0; i < pdat->pads.size() / 2; i++) {
			pad = (ceilf(x->dims[i + 2] / (float)pdat->strides[i]) - 1) * pdat->strides[i] + ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) - x->dims[i + 2];
			pdat->cpads[i + pdat->kernels.size()] = pad / 2;
			pdat->cpads[i] = pad - pdat->cpads[i + pdat->kernels.size()];
		}
		break;
	case AUTO_PAD_VALID:
		memset(pdat->cpads, 0, sizeof(int) * pdat->pads.size());
		break;
	default:
		break;
	}
	dims[0] = x->dims[0];
	dims[1] = w->dims[0];
	for (i = 0; i < ndim - 2; i++) {
		switch (pdat->auto_pad) {
		case AUTO_PAD_NOTSET:
			dims[i + 2] = floorf((x->dims[i + 2] + pdat->cpads[i] + pdat->cpads[i + pdat->kernels.size()] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1)) / (float)pdat->strides[i] + 1);
			break;
		case AUTO_PAD_SAME_UPPER:
		case AUTO_PAD_SAME_LOWER:
			dims[i + 2] = ceilf(x->dims[i + 2] / (float)pdat->strides[i]);
			break;
		case AUTO_PAD_VALID:
			dims[i + 2] = ceilf((x->dims[i + 2] - ((pdat->kernels[i] - 1) * pdat->dilations[i] + 1) + 1) / (float)pdat->strides[i]);
			break;
		default:
			break;
		}
	}
	return y->reshape(&dims[0], ndim, x->type);
}

template <typename T>
inline void dgemm_generic(int n, int m, int o, T* A, T* B, T* C)
{
	ref2d<T> atype_A(o, A);
	ref2d<T> btype_B(m, B);
	ref2d<T> ctype_C(m, C);
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			ctype_C[i][j] = 0.;
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < o; ++k) {
			for (int j = 0; j < m; ++j) {
				ctype_C[i][j] += atype_A[i][k] * btype_B[k][j];
			}
		}
	}
}

template <typename T>
void Conv_generic(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* w = n->inputs[1];
	onnx_tensor_t* b = nullptr;
	T* pb = nullptr;

	conv_mode_t conv_mode = CONV_SIMPLE;
	T* pxcache = nullptr;
	T* matw = nullptr;
	T* matx = nullptr;
	T* maty = nullptr;

	T sum, v, weight;
	int ndim = x->ndim;
	int M = w->dims[0];
	int C = w->dims[1];
	int H = w->dims[2];
	int W = w->dims[3];
	int ch, i;

	if (n->inputs.size() > 2) {
		b = n->inputs[2];
		pb = (T*)b->data;
	}
	if (ndim == 4) {
		int iC = x->dims[1];
		int iH = x->dims[2];
		int iW = x->dims[3];

		int oN = y->dims[0];
		int oC = w->dims[0];
		int oH = y->dims[2];
		int oW = y->dims[3];

		int MM = M / pdat->group;
		int CC = iC / pdat->group;

		ref4d<T> pxcachetype_pxcache(W, H, (oC * pdat->group / M) * C);
		ref2d<T> mwtype_matw(/*[H * W * C]*/MM);
		ref2d<T> mxtype_matx(/*[oH * oW]*/H * W * C);
		ref2d<T> mytype_maty(/*[oH * oW]*/MM);

		ref4d<T> px(iW, iH, iC, (T*)x->data);
		ref4d<T> py(oW, oH, M, (T*)y->data);
		ref4d<T> pw(W, H, C, (T*)w->data);

		/* try im2col first */
		matw = new (std::nothrow) T[MM * H * W * C];
		matx = new (std::nothrow) T[oH * oW * H * W * C];
		maty = new (std::nothrow) T[oH * oW * MM];

		mwtype_matw() = matw;
		mxtype_matx() = matx;
		mytype_maty() = maty;
		if (matw && matx && maty) {
			conv_mode = CONV_IM2COL;
		}else {
			if (matw) delete matw;
			if (matx) delete matx;
			if (maty) delete maty;
			
			/* then try cached conv */
			pxcache = new (std::nothrow) T[oN * (oC * pdat->group / M) * C * H * W];
			if (pxcache) {
				conv_mode = CONV_CACHED;
			}
		}

		if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED) {
			pxcachetype_pxcache() = pxcache;
			for (int h = 0; h < oH; ++h) {
				for (int w = 0; w < oW; ++w) {
					int base_h = h * pdat->strides[0] - pdat->cpads[0];
					int base_w = w * pdat->strides[1] - pdat->cpads[1];

					if (pxcache) {
						for (int n = 0; n < oN; ++n) {
							for (int group_c = 0; group_c < oC * pdat->group / M; ++group_c) {
								int base_c = group_c * C;
								for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
									int input_h = base_h + i * pdat->dilations[0];
									if (input_h >= iH)
										break;
									for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
										int input_w = base_w + j * pdat->dilations[1];
										if (input_w >= iW)
											break;
										for (int w_channel = 0; w_channel < C; ++w_channel) {
											ch = base_c + w_channel;
											pxcachetype_pxcache[n][ch][i][j] = px[n][ch][input_h][input_w];
										}
									}
								}
							}
						}
					}

					for (int n = 0; n < oN; ++n) {
						for (int c = 0; c < oC; ++c) {
							int base_c = (c * pdat->group / M) * C;
							sum = 0;
							for (int i = (base_h < 0 ? (-base_h) / pdat->dilations[0] : 0); i < H; ++i) {
								int input_h = base_h + i * pdat->dilations[0];
								if (input_h >= iH)
									break;
								for (int j = (base_w < 0 ? (-base_w) / pdat->dilations[1] : 0); j < W; ++j) {
									int input_w = base_w + j * pdat->dilations[1];
									if (input_w >= iW)
										break;
									for (int w_channel = 0; w_channel < C; ++w_channel) {
										ch = base_c + w_channel;
										if (pxcache) {										
											v = pxcachetype_pxcache[n][ch][i][j];
										}else {
											v = px[n][ch][input_h][input_w];
										}									
										weight = pw[c][w_channel][i][j];
										sum += v * weight;
									}
								}
							}
							if (pb)
								sum += pb[c];
							py[n][c][h][w] = sum;
						}
					}
				}
			}
			if (pxcache) {
				delete pxcache;
			}
		}else if (conv_mode == CONV_IM2COL) {			
			for (int g = 0; g < pdat->group; g++) {
				for (size_t m = 0; m < MM; m++) {
					for (size_t c = 0; c < C; c++) {
						for (size_t h = 0; h < H; h++) {
							for (size_t w = 0; w < W; w++) {
								mwtype_matw[c * H * W + h * W + w][m] = pw[g * MM + m][c][h][w];
							}						
						}					
					}				
				}
				
				for (int n = 0; n < oN; n++) {
					for (size_t hh = 0; hh < oH; hh++) {
						for (size_t ww = 0; ww < oW; ww++) {
							int base_h = hh * pdat->strides[0] - pdat->cpads[0];
							int base_w = ww * pdat->strides[1] - pdat->cpads[1];
							for (size_t c = 0; c < C; c++) {
								for (size_t h = 0; h < H; h++) {
									for (size_t w = 0; w < W; w++) {
										int ih = base_h + h * pdat->dilations[0];
										int iw = base_w + w * pdat->dilations[1];
										if (ih < 0 || iw < 0 || ih >= iH || iw >= iW) {
											mxtype_matx[hh * oW + ww][c * H * W + h * W + w] = 0.;
										}else {	
											mxtype_matx[hh * oW + ww][c * H * W + h * W + w] = px[n][g * CC + c][ih][iw];
										}
									}
								}
							}
						}
					}
					dgemm_generic(oH * oW, MM, H * W * C, matx, matw, maty);
					for (int m = 0; m < MM; ++m) {
						for (int h = 0; h < oH; ++h) {
							for (int w = 0; w < oW; ++w) {
								T t = mytype_maty[h * oW + w][m];
								if (pb) {
									t += pb[g * MM + m];
								}
								py[n][g * MM + m][h][w] = t;
							}
						}
					}
				}
			}
			delete matw;
			delete matx;
			delete maty;
		}else {
			/* never */
		}
	}else {
		T* px = (T*)x->data;
		T* py = (T*)y->data;
		T* pw = (T*)w->data;

		std::vector<int> i_dim(ndim);
		std::vector<int> o_dim(ndim);
		std::vector<int> w_dim(ndim);
		std::vector<int> b_dim(ndim);

		memset(&o_dim[0], 0, sizeof(o_dim));
		do {
			b_dim[0] = o_dim[0];
			for (i = 2; i < ndim; i++)
				b_dim[i] = o_dim[i] * pdat->strides[i - 2] - pdat->cpads[i - 2];
			sum = 0;
			memset(&w_dim[0], 0, sizeof(w_dim));
			w_dim[0] = o_dim[1];
			do {
				if (w_dim[1] == 1)
					break;
				i_dim[0] = b_dim[0];
				for (i = 2; i < ndim; i++)
					i_dim[i] = b_dim[i] + w_dim[i] * pdat->dilations[i - 2];
				for (ch = 0; ch < C; ch++) {
					i_dim[1] = (o_dim[1] * pdat->group / M) * C + ch;
					w_dim[1] = ch;
					for (i = 0; i < ndim; i++) {
						if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
							v = 0;
							break;
						}
					}
					if (i >= ndim)
						v = px[dim_offset(ndim, &i_dim[0], &x->dims[0])];
					for (i = 0; i < ndim; i++) {
						if ((w_dim[i] < 0) || (w_dim[i] >= w->dims[i])) {
							weight = 0;
							break;
						}
					}
					if (i >= ndim)
						weight = pw[dim_offset(ndim, &w_dim[0], &w->dims[0])];
					sum += v * weight;
				}
				w_dim[1] = 0;
			} while (dim_next(ndim, &w_dim[0], &w->dims[0]));
			if (pb)
				sum += pb[o_dim[1]];
			py[dim_offset(ndim, &o_dim[0], &y->dims[0])] = sum;
		} while (dim_next(ndim, &o_dim[0], &y->dims[0]));
	}
}

GEN_HOLEDR_TYPE(holder, Conv_generic)

} // namespace

void resolver_default_op_Conv(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Conv_init;
		n->reshape = Conv_reshape;
	}
}
