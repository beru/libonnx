#include "onnx.h"
#include "refnd.h"
#include "util.h"

namespace onnx {

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

struct Conv_operator : public operator_t {
	auto_pad_t auto_pad = AUTO_PAD_NOTSET;
	int group = 0;
	std::vector<int> kernels;
	std::vector<int> dilations;
	std::vector<int> pads;
	std::vector<int> strides;

	int cpads[32] = {0};

	bool init() override {
		if (!(inputs.size() >= 2 && outputs.size() == 1)) {
			return false;
		}
		int64_t* ints = nullptr;
		int i, l;
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
		group = attribute("group", 1);
		int nkernel = attribute("kernel_shape", ints);
		if (nkernel > 0) {
			kernels.resize(nkernel);
			for (i=0; i<nkernel; ++i) {
				kernels[i] = ints[i];
			}
			int ndilation = nkernel;
			dilations.resize(ndilation);
			l = attribute("dilations", ints);
			for (i = 0; i < l; ++i) {
				dilations[i] = ints[i];
			}
			for (; i < ndilation; ++i) {
				dilations[i] = 1;
			}
			int npad = nkernel * 2;
			pads.resize(npad);
			l = attribute("pads", ints);
			for (i = 0; i < l; ++i) {
				pads[i] = ints[i];
			}
			for (; i < npad; ++i) {
				pads[i] = 0;
			}
			int nstride = nkernel;
			strides.resize(nstride);
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
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* w = inputs[1];
		const int ndim = x->ndim;
		std::vector<int> dims(ndim);

		switch (auto_pad) {
		case AUTO_PAD_NOTSET:
			memcpy(cpads, &pads[0], sizeof(int) * pads.size());
			break;
		case AUTO_PAD_SAME_UPPER:
			for (int i = 0; i < pads.size() / 2; ++i) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
				cpads[i] = pad / 2;
				cpads[i + kernels.size()] = pad - cpads[i];
			}
			break;
		case AUTO_PAD_SAME_LOWER:
			for (int i = 0; i < pads.size() / 2; ++i) {
				int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
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
		dims[1] = w->dims[0];
		for (int i = 0; i < ndim - 2; ++i) {
			switch (auto_pad) {
			case AUTO_PAD_NOTSET:
				dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
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
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* w = inputs[1];
		const tensor_t* b = nullptr;
		T* pb = nullptr;

		conv_mode_t conv_mode = CONV_SIMPLE;
		T* pxcache = nullptr;
		T* matw = nullptr;
		T* matx = nullptr;
		T* maty = nullptr;

		T sum, v, weight;
		const int ndim = x->ndim;
		int M = w->dims[0];
		int C = w->dims[1];
		int H = w->dims[2];
		int W = w->dims[3];
		int ch, i;

		if (inputs.size() > 2) {
			b = inputs[2];
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

			int MM = M / group;
			int CC = iC / group;

			ref4d<T> pxcachetype_pxcache(W, H, (oC * group / M) * C);
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
				pxcache = new (std::nothrow) T[oN * (oC * group / M) * C * H * W];
				if (pxcache) {
					conv_mode = CONV_CACHED;
				}
			}

			if (conv_mode == CONV_SIMPLE || conv_mode == CONV_CACHED) {
				pxcachetype_pxcache() = pxcache;
				for (int h = 0; h < oH; ++h) {
					for (int w = 0; w < oW; ++w) {
						int base_h = h * strides[0] - cpads[0];
						int base_w = w * strides[1] - cpads[1];

						if (pxcache) {
							for (int n = 0; n < oN; ++n) {
								for (int group_c = 0; group_c < oC * group / M; ++group_c) {
									int base_c = group_c * C;
									for (int i = (base_h < 0 ? (-base_h) / dilations[0] : 0); i < H; ++i) {
										int input_h = base_h + i * dilations[0];
										if (input_h >= iH) {
											break;
										}
										for (int j = (base_w < 0 ? (-base_w) / dilations[1] : 0); j < W; ++j) {
											int input_w = base_w + j * dilations[1];
											if (input_w >= iW) {
												break;
											}
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
								int base_c = (c * group / M) * C;
								sum = 0;
								for (int i = (base_h < 0 ? (-base_h) / dilations[0] : 0); i < H; ++i) {
									int input_h = base_h + i * dilations[0];
									if (input_h >= iH) {
										break;
									}
									for (int j = (base_w < 0 ? (-base_w) / dilations[1] : 0); j < W; ++j) {
										int input_w = base_w + j * dilations[1];
										if (input_w >= iW) {
											break;
										}
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
								if (pb) {
									sum += pb[c];
								}
								py[n][c][h][w] = sum;
							}
						}
					}
				}
				if (pxcache) {
					delete pxcache;
				}
			}else if (conv_mode == CONV_IM2COL) {			
				for (int g = 0; g < group; ++g) {
					for (size_t m = 0; m < MM; ++m) {
						for (size_t c = 0; c < C; ++c) {
							for (size_t h = 0; h < H; ++h) {
								for (size_t w = 0; w < W; ++w) {
									mwtype_matw[c * H * W + h * W + w][m] = pw[g * MM + m][c][h][w];
								}						
							}					
						}				
					}
				
					for (int n = 0; n < oN; ++n) {
						for (size_t hh = 0; hh < oH; ++hh) {
							for (size_t ww = 0; ww < oW; ++ww) {
								int base_h = hh * strides[0] - cpads[0];
								int base_w = ww * strides[1] - cpads[1];
								for (size_t c = 0; c < C; ++c) {
									for (size_t h = 0; h < H; ++h) {
										for (size_t w = 0; w < W; ++w) {
											int ih = base_h + h * dilations[0];
											int iw = base_w + w * dilations[1];
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
			const T* px = (const T*)x->data;
			T* py = (T*)y->data;
			T* pw = (T*)w->data;

			std::vector<int> i_dim(ndim);
			std::vector<int> o_dim(ndim);
			std::vector<int> w_dim(ndim);
			std::vector<int> b_dim(ndim);

			do {
				b_dim[0] = o_dim[0];
				for (i = 2; i < ndim; ++i) {
					b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
				}
				sum = 0;
				std::fill(w_dim.begin(), w_dim.end(), 0);
				w_dim[0] = o_dim[1];
				do {
					if (w_dim[1] == 1) {
						break;
					}
					i_dim[0] = b_dim[0];
					for (i = 2; i < ndim; ++i) {
						i_dim[i] = b_dim[i] + w_dim[i] * dilations[i - 2];
					}
					for (ch = 0; ch < C; ++ch) {
						i_dim[1] = (o_dim[1] * group / M) * C + ch;
						w_dim[1] = ch;
						for (i = 0; i < ndim; ++i) {
							if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
								v = 0;
								break;
							}
						}
						if (i >= ndim) {
							v = px[dim_offset(ndim, &i_dim[0], &x->dims[0])];
						}
						for (i = 0; i < ndim; ++i) {
							if ((w_dim[i] < 0) || (w_dim[i] >= w->dims[i])) {
								weight = 0;
								break;
							}
						}
						if (i >= ndim) {
							weight = pw[dim_offset(ndim, &w_dim[0], &w->dims[0])];
						}
						sum += v * weight;
					}
					w_dim[1] = 0;
				} while (dim_next(ndim, &w_dim[0], &w->dims[0]));
				if (pb) {
					sum += pb[o_dim[1]];
				}
				py[dim_offset(ndim, &o_dim[0], &y->dims[0])] = sum;
			} while (dim_next(ndim, &o_dim[0], &y->dims[0]));
		}
	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 11) {
			typed_exec<Conv_operator,
				float16_t, float, double
			>(this, type);
		}else if (opset >= 1) {
			typed_exec<Conv_operator,
				float16_t, float, double
			>(this, type);
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Conv(int opset)
{
	return new Conv_operator;
}

} // namespace onnx
