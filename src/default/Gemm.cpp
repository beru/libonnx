#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Gemm_operator : public operator_t {
	float alpha;
	float beta;
	int transA;
	int transB;

	int m = 0;
	int n = 0;
	int k = 0;

	bool init() override {
		if (!(inputs.size() >= 2 && outputs.size() == 1)) {
			return false;
		}
		alpha = attribute("alpha", 1.0f);
		beta = attribute("beta", 1.0f);
		transA = attribute("transA", 0);
		transB = attribute("transB", 0);
		return true;
	}

	bool reshape() override {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
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
			return false;
		if (m <= 0 || n <= 0 || k <= 0)
			return false;
		int tmp[2] = { m, n };
		if ((inputs.size() > 2) && !inputs[2]->broadcast_is_valid(tmp, 2))
			return false;
		return y->reshape(tmp, 2, a->type);
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		const tensor_t* a = inputs[0];
		const tensor_t* b = inputs[1];
		const tensor_t* c = (inputs.size() > 2) ? inputs[2] : nullptr;
		T* py = (T*)y->data;
		const T* pa = (T*)a->data;
		const T* pb = (T*)b->data;
		const T* pc;
		T sum;
		int oa = 0;
		int ob = 0;
		int oy = 0;

		if (transA && transB) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					sum = 0;
					for (int k = 0; k < k; k++) {
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
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					sum = 0;
					for (int k = 0; k < k; k++) {
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
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					sum = 0;
					for (int k = 0; k < k; k++) {
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
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					sum = 0;
					for (int k = 0; k < k; k++) {
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

	void exec() override {
		auto input_type = inputs[0]->type;
		if (opset >= 13) {
			TYPED_EXEC(input_type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			)
		}else if (opset >= 11) {
			TYPED_EXEC(input_type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 9) {
			TYPED_EXEC(input_type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (opset >= 7) {
			TYPED_EXEC(input_type,
				float16_t, float, double
			)
		}else if (opset >= 6) {
		}else if (opset >= 1) {
		}
	}
};

} // namespace {

operator_t* resolver_default_op_Gemm()
{
	return new Gemm_operator;
}

} // namespace onnx
