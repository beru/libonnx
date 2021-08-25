#include <onnx.h>
#include "util.h"

namespace {

struct ope_pdata_t {
	onnx_tensor_type_t to;
};

bool Cast_init(onnx_node_t* n)
{
	if (!is_inout_size(n, 1, 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->to = (onnx_tensor_type_t)n->attribute_read_int("to", n->inputs[0]->type);
	n->priv = pdat;
	return true;
}

int Cast_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int Cast_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, pdat->to);
}

void Cast_bool(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1.0 : 0.0;
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%u", (px[i] != 0) ? 1 : 0);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_int8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int8_t* px = (int8_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_int16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_int32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int32_t* px = (int32_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%d", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_int64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t* px = (int64_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%lld", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_uint8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_uint16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_uint32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint32_t* px = (uint32_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%u", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_uint64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint64_t* px = (uint64_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%llu", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_bfloat16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%g", (float)px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%g", (float)px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%g", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (px[i] != 0.0) ? 1 : 0;
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = px[i];
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			char buf[32];
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				sprintf(buf, "%g", px[i]);
				py[i] = strdup(buf);
			}
		}
		break;
	default:
		break;
	}
}

void Cast_string(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	char** px = (char**)x->data;
	size_t i, l;

	switch (pdat->to) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int8_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int16_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int32_t)strtol(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (int64_t)strtoll(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint8_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint16_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint32_t)strtoul(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (uint64_t)strtoull(px[i], 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)strtod(px[i], nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)strtod(px[i], nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (float)strtod(px[i], nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)y->data;
			for (i = 0, l = y->ndata; i < l; i++)
				py[i] = (double)strtod(px[i], nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char** py = (char**)y->data;
			for (i = 0, l = y->ndata; i < l; i++) {
				if (py[i])
					free(py[i]);
				py[i] = strdup(px[i]);
			}
		}
		break;
	default:
		break;
	}
}

} // namespace

void resolver_default_op_Cast(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Cast_bool,
			.int8_ = Cast_int8,
			.int16_ = Cast_int16,
			.int32_ = Cast_int32,
			.int64_ = Cast_int64,
			.uint8_ = Cast_uint8,
			.uint16_ = Cast_uint16,
			.uint32_ = Cast_uint32,
			.uint64_ = Cast_uint64,
			.bfloat16_ = Cast_bfloat16,
			.float16_ = Cast_float16,
			.float32_ = Cast_float32,
			.float64_ = Cast_float64,
			.string_ = Cast_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Cast_bool,
			.int8_ = Cast_int8,
			.int16_ = Cast_int16,
			.int32_ = Cast_int32,
			.int64_ = Cast_int64,
			.uint8_ = Cast_uint8,
			.uint16_ = Cast_uint16,
			.uint32_ = Cast_uint32,
			.uint64_ = Cast_uint64,
			.float16_ = Cast_float16,
			.float32_ = Cast_float32,
			.float64_ = Cast_float64,
			.string_ = Cast_string,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.bool_ = Cast_bool,
			.int8_ = Cast_int8,
			.int16_ = Cast_int16,
			.int32_ = Cast_int32,
			.int64_ = Cast_int64,
			.uint8_ = Cast_uint8,
			.uint16_ = Cast_uint16,
			.uint32_ = Cast_uint32,
			.uint64_ = Cast_uint64,
			.float16_ = Cast_float16,
			.float32_ = Cast_float32,
			.float64_ = Cast_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Cast_init;
		n->exit = Cast_exit;
		n->reshape = Cast_reshape;
	}
}
