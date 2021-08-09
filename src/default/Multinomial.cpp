#include <onnx.h>

struct operator_pdata_t {
	onnx_tensor_type_t dtype;
	int sample_size;
	float seed;
};

static int Multinomial_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->dtype = (onnx_tensor_type_t)n->attribute_read_int("dtype", 6);
		pdat->sample_size = n->attribute_read_int("sample_size", 1);
		pdat->seed = n->attribute_read_float("seed", 0.0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Multinomial_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Multinomial_reshape(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, pdat->dtype);
}

static void Multinomial_float16(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	uint16_t* px = (uint16_t*)x->datas;
	std::vector<float> cum(csz);
	int i, j, k, l, o;

	if (pdat->seed != 0.0)
		srand(pdat->seed);

	switch (y->type) {
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = float16_to_float32(px[i * csz]);
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + float16_to_float32(px[i * csz + k]);
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((float)rand() / (float)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = float16_to_float32(px[i * csz]);
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + float16_to_float32(px[i * csz + k]);
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((float)rand() / (float)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

static void Multinomial_float32(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	float* px = (float*)x->datas;
	std::vector<float> cum(csz);
	int i, j, k, l, o;

	if (pdat->seed != 0.0)
		srand(pdat->seed);

	switch (y->type) {
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((float)rand() / (float)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((float)rand() / (float)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

static void Multinomial_float64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int bsz = x->dims[0];
	int csz = x->dims[1];
	double* px = (double*)x->datas;
	std::vector<double> cum(csz);
	int i, j, k, l, o;

	if (pdat->seed != 0.0)
		srand(pdat->seed);

	switch (y->type) {
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((double)rand() / (double)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->datas;
			for (i = 0; i < bsz; i++) {
				for (j = 0; j < pdat->sample_size; j++) {
					cum[0] = px[i * csz];
					for (k = 1; k < csz; k++)
						cum[k] = cum[k - 1] + px[i * csz + k];
					for (k = 0, l = csz - 1; k < csz - 1; k++) {
						if ((double)rand() / (double)(RAND_MAX) < cum[k]) {
							l = k;
							break;
						}
					}
					o = i * csz + l;
					py[o]++;
				}
			}
		}
		break;
	default:
		break;
	}
}

void resolver_default_op_Multinomial(onnx_node_t* n)
{
	if (n->opset >= 7) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->ope = Multinomial_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->ope = Multinomial_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->ope = Multinomial_float64;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Multinomial_init;
		n->exit = Multinomial_exit;
		n->reshape = Multinomial_reshape;
	}
}
