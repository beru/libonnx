#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Gemm_operator : public operator_t {
	float alpha;
	float beta;
	int transA;
	int transB;

	int m = 0;
	int n = 0;
	int k = 0;

	bool init() override {
		if (!(operator_t::n->inputs.size() >= 2 && operator_t::n->outputs.size() == 1)) {
			return false;
		}
		alpha = operator_t::n->attribute("alpha", 1.0f);
		beta = operator_t::n->attribute("beta", 1.0f);
		transA = operator_t::n->attribute("transA", 0);
		transB = operator_t::n->attribute("transB", 0);
		return true;
	}

	bool reshape() override {
		tensor_t* y = operator_t::n->outputs[0];
		const tensor_t* a = operator_t::n->inputs[0];
		const tensor_t* b = operator_t::n->inputs[1];
		int k;

		if (transA) {
			m = a->dims[1];
			k = a->dims[0];
		}else {
			m = a->dims[0];
			k = a->dims[1];
		}
		if (transB) {
			n = b->dims[0];
			k = 1;
		}else {
			n = b->dims[1];
			k = 0;
		}
		if (b->dims[k] != k)
			return 0;
		if (m <= 0 || n <= 0 || k <= 0)
			return 0;
		int tmp[2] = { m, n };
		if ((operator_t::n->inputs.size() > 2) && !operator_t::n->inputs[2]->broadcast_is_valid(tmp, 2))
			return 0;
		return y->reshape(tmp, 2, a->type);
	}

	void exec() override {
		tensor_t* y = operator_t::n->outputs[0];
		const tensor_t* a = operator_t::n->inputs[0];
		const tensor_t* b = operator_t::n->inputs[1];
		const tensor_t* c = (operator_t::n->inputs.size() > 2) ? operator_t::n->inputs[2] : nullptr;
		T* py = (T*)y->data;
		const T* pa = (T*)a->data;
		const T* pb = (T*)b->data;
		const T* pc;
		T sum;
		int oa = 0;
		int ob = 0;
		int oy = 0;
		int i, j, k;

		if (transA && transB) {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					sum = 0;
					for (k = 0; k < k; k++) {
						sum += pa[oa] * pb[ob];
						oa += m;
						ob += 1;
					}
					oa -= m * k;
					ob -= k;
					if (c) {
						pc = (const T*)c->broadcast_map_address(y, oy);
						py[oy] = alpha * sum + beta * (*pc);
					}
					else
						py[oy] = alpha * sum;
					oy++;
					ob += k;
				}
				ob -= n * k;
				oa++;
			}
		}else if (transA) {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					sum = 0;
					for (k = 0; k < k; k++) {
						sum += pa[oa] * pb[ob];
						oa += m;
						ob += n;
					}
					oa -= m * k;
					ob -= n * k;
					if (c) {
						pc = (const T*)c->broadcast_map_address(y, oy);
						py[oy] = alpha * sum + beta * (*pc);
					}
					else
						py[oy] = alpha * sum;
					oy++;
					ob++;
				}
				ob -= n;
				oa++;
			}
		}else if (transB) {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					sum = 0;
					for (k = 0; k < k; k++) {
						sum += pa[oa] * pb[ob];
						oa += 1;
						ob += 1;
					}
					oa -= k;
					ob -= k;
					if (c) {
						pc = (const T*)c->broadcast_map_address(y, oy);
						py[oy] = alpha * sum + beta * (*pc);
					}
					else
						py[oy] = alpha * sum;
					oy++;
					ob += k;
				}
				ob -= n * k;
				oa += k;
			}
		}else {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					sum = 0;
					for (k = 0; k < k; k++) {
						sum += pa[oa] * pb[ob];
						oa += 1;
						ob += n;
					}
					oa -= k;
					ob -= n * k;
					if (c) {
						pc = (const T*)c->broadcast_map_address(y, oy);
						py[oy] = alpha * sum + beta * (*pc);
					}
					else
						py[oy] = alpha * sum;
					oy++;
					ob++;
				}
				ob -= n;
				oa += k;
			}
		}
	}
};

void resolver_default_op_Gemm(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Gemm_operator,
			int32_t, int64_t,
			uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = ope_type_select<Gemm_operator,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<Gemm_operator,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<Gemm_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
