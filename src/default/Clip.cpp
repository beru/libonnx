#include <onnx.h>
#include "util.h"

namespace {

union onnx_scalar_t {
	uint8_t v_bool;
	int8_t v_int8;
	int16_t v_int16;
	int32_t v_int32;
	int64_t v_int64;
	uint8_t v_uint8;
	uint16_t v_uint16;
	uint32_t v_uint32;
	uint64_t v_uint64;
	uint16_t v_bfloat16;
	uint16_t v_float16;
	float v_float32;
	double v_float64;
	struct {
		float real;
		float imaginary;
	} v_complex64;
	struct {
		double real;
		double imaginary;
	} v_complex128;
};

struct ope_pdata_t {
	onnx_scalar_t* pmin;
	onnx_scalar_t* pmax;
};

bool Clip_init(onnx_node_t* n)
{
	if (!(n->inputs.size() >= 1 && n->outputs.size() == 1)) {
		return false;
	}
	ope_pdata_t* pdat = new (std::nothrow) ope_pdata_t;
	if (!pdat)
		return false;
	pdat->pmin = nullptr;
	pdat->pmax = nullptr;
	n->priv = pdat;
	return true;
}

int Clip_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

int Clip_reshape(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	pdat->pmin = nullptr;
	pdat->pmax = nullptr;
	for (int i = 1; i < min<size_t>(3, n->inputs.size()); i++) {
		if (n->inputs[i]->ndim == 0) {
			if (n->inputs[i]->name == "min")
				pdat->pmin = (onnx_scalar_t*)n->inputs[i]->data;
			else if (n->inputs[i]->name == "max")
				pdat->pmax = (onnx_scalar_t*)n->inputs[i]->data;
		}
	}
	return y->reshape_identity(x);
}

void Clip_int8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->data;
	int16_t* py = (int16_t*)y->data;
	int8_t minv = pdat->pmin ? pdat->pmin->v_int8 : INT8_MIN;
	int8_t maxv = pdat->pmax ? pdat->pmax->v_int8 : INT8_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_int16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int16_t* px = (int16_t*)x->data;
	int16_t* py = (int16_t*)y->data;
	int16_t minv = pdat->pmin ? pdat->pmin->v_int16 : INT16_MIN;
	int16_t maxv = pdat->pmax ? pdat->pmax->v_int16 : INT16_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_int32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int32_t* px = (int32_t*)x->data;
	int32_t* py = (int32_t*)y->data;
	int32_t minv = pdat->pmin ? pdat->pmin->v_int32 : INT32_MIN;
	int32_t maxv = pdat->pmax ? pdat->pmax->v_int32 : INT32_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_int64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	int64_t* px = (int64_t*)x->data;
	int64_t* py = (int64_t*)y->data;
	int64_t minv = pdat->pmin ? pdat->pmin->v_int64 : INT64_MIN;
	int64_t maxv = pdat->pmax ? pdat->pmax->v_int64 : INT64_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_uint8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->data;
	uint8_t* py = (uint8_t*)y->data;
	uint8_t minv = pdat->pmin ? pdat->pmin->v_uint8 : 0;
	uint8_t maxv = pdat->pmax ? pdat->pmax->v_uint8 : UINT8_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_uint16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	uint16_t* py = (uint16_t*)y->data;
	uint16_t minv = pdat->pmin ? pdat->pmin->v_uint16 : 0;
	uint16_t maxv = pdat->pmax ? pdat->pmax->v_uint16 : UINT16_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_uint32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint32_t* px = (uint32_t*)x->data;
	uint32_t* py = (uint32_t*)y->data;
	uint32_t minv = pdat->pmin ? pdat->pmin->v_uint32 : 0;
	uint32_t maxv = pdat->pmax ? pdat->pmax->v_uint32 : UINT32_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_uint64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint64_t* px = (uint64_t*)x->data;
	uint64_t* py = (uint64_t*)y->data;
	uint64_t minv = pdat->pmin ? pdat->pmin->v_uint64 : 0;
	uint64_t maxv = pdat->pmax ? pdat->pmax->v_uint64 : UINT64_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_bfloat16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	uint16_t* py = (uint16_t*)y->data;
	float minv = bfloat16_to_float32(pdat->pmin ? pdat->pmin->v_bfloat16 : float32_to_bfloat16(FLT_MIN));
	float maxv = bfloat16_to_float32(pdat->pmax ? pdat->pmax->v_bfloat16 : float32_to_bfloat16(FLT_MAX));
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = bfloat16_to_float32(px[i]);
		if (v < minv)
			v = minv;
		else if (px[i] > maxv)
			v = maxv;
		else
			v = px[i];
		py[i] = float32_to_bfloat16(v);
	}
}

void Clip_float16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint16_t* px = (uint16_t*)x->data;
	uint16_t* py = (uint16_t*)y->data;
	float minv = float16_to_float32(pdat->pmin ? pdat->pmin->v_float16 : float32_to_float16(FLT_MIN));
	float maxv = float16_to_float32(pdat->pmax ? pdat->pmax->v_float16 : float32_to_float16(FLT_MAX));
	float v;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		v = float16_to_float32(px[i]);
		if (v < minv)
			v = minv;
		else if (px[i] > maxv)
			v = maxv;
		else
			v = px[i];
		py[i] = float32_to_float16(v);
	}
}

void Clip_float32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	float* px = (float*)x->data;
	float* py = (float*)y->data;
	float minv = pdat->pmin ? pdat->pmin->v_float32 : FLT_MIN;
	float maxv = pdat->pmax ? pdat->pmax->v_float32 : FLT_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

void Clip_float64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	double* px = (double*)x->data;
	double* py = (double*)y->data;
	double minv = pdat->pmin ? pdat->pmin->v_float64 : DBL_MIN;
	double maxv = pdat->pmax ? pdat->pmax->v_float64 : DBL_MAX;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < minv)
			py[i] = minv;
		else if (px[i] > maxv)
			py[i] = maxv;
		else
			py[i] = px[i];
	}
}

} // namespace

void resolver_default_op_Clip(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Clip_int8,
			.int16_ = Clip_int16,
			.int32_ = Clip_int32,
			.int64_ = Clip_int64,
			.uint8_ = Clip_uint8,
			.uint16_ = Clip_uint16,
			.uint32_ = Clip_uint32,
			.uint64_ = Clip_uint64,
			.bfloat16_ = Clip_bfloat16,
			.float16_ = Clip_float16,
			.float32_ = Clip_float32,
			.float64_ = Clip_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Clip_int8,
			.int16_ = Clip_int16,
			.int32_ = Clip_int32,
			.int64_ = Clip_int64,
			.uint8_ = Clip_uint8,
			.uint16_ = Clip_uint16,
			.uint32_ = Clip_uint32,
			.uint64_ = Clip_uint64,
			.float16_ = Clip_float16,
			.float32_ = Clip_float32,
			.float64_ = Clip_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Clip_float16,
			.float32_ = Clip_float32,
			.float64_ = Clip_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Clip_init;
		n->exit = Clip_exit;
		n->reshape = Clip_reshape;
	}

}
