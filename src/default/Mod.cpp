#include <onnx.h>

struct operator_pdata_t {
	int fmod;
};

static int Mod_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1)) {
		operator_pdata_t* pdat = new operator_pdata_t;
		pdat->fmod = n->attribute_read_int("fmod", 0);
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int Mod_exit(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int Mod_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
static void Mod_int(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa;
	T* pb;
	T t;

	if (pdat->fmod) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (T*)a->broadcast_map_address(y, i);
			pb = (T*)b->broadcast_map_address(y, i);
			py[i] = fmodf(*pa, *pb);
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (T*)a->broadcast_map_address(y, i);
			pb = (T*)b->broadcast_map_address(y, i);
			t = *pa % *pb;
			if (((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0)))
				t += *pb;
			py[i] = t;
		}
	}
}

static void Mod_int64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	int64_t* py = (int64_t*)y->datas;
	int64_t* pa;
	int64_t* pb;
	int64_t t;

	if (pdat->fmod) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (int64_t*)a->broadcast_map_address(y, i);
			pb = (int64_t*)b->broadcast_map_address(y, i);
			py[i] = fmod(*pa, *pb);
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pb = (int64_t*)b->broadcast_map_address(y, i);
			pa = (int64_t*)a->broadcast_map_address(y, i);
			t = *pa % *pb;
			if (((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0)))
				t += *pb;
			py[i] = t;
		}
	}
}

template <typename T>
static void Mod_uint(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa;
	T* pb;

	if (pdat->fmod) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (T*)a->broadcast_map_address(y, i);
			pb = (T*)b->broadcast_map_address(y, i);
			py[i] = fmodf(*pa, *pb);
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (T*)a->broadcast_map_address(y, i);
			pb = (T*)b->broadcast_map_address(y, i);
			py[i] = *pa % *pb;
		}
	}
}

static void Mod_uint64(onnx_node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint64_t* py = (uint64_t*)y->datas;
	uint64_t* pa;
	uint64_t* pb;

	if (pdat->fmod) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint64_t*)a->broadcast_map_address(y, i);
			pb = (uint64_t*)b->broadcast_map_address(y, i);
			py[i] = fmod(*pa, *pb);
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint64_t*)a->broadcast_map_address(y, i);
			pb = (uint64_t*)b->broadcast_map_address(y, i);
			py[i] = *pa % *pb;
		}
	}
}

static void Mod_bfloat16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = float32_to_bfloat16(fmodf(bfloat16_to_float32(*pa), bfloat16_to_float32(*pb)));
	}
}

static void Mod_float16(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (uint16_t*)a->broadcast_map_address(y, i);
		pb = (uint16_t*)b->broadcast_map_address(y, i);
		py[i] = float32_to_float16(fmodf(float16_to_float32(*pa), float16_to_float32(*pb)));
	}
}

static void Mod_float32(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	float* py = (float*)y->datas;
	float* pa;
	float* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (float*)a->broadcast_map_address(y, i);
		pb = (float*)b->broadcast_map_address(y, i);
		py[i] = fmodf(*pa, *pb);
	}
}

static void Mod_float64(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	double* py = (double*)y->datas;
	double* pa;
	double* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (double*)a->broadcast_map_address(y, i);
		pb = (double*)b->broadcast_map_address(y, i);
		py[i] = fmod(*pa, *pb);
	}
}

void resolver_default_op_Mod(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Mod_int<int8_t>,
			.int16_ = Mod_int<int16_t>,
			.int32_ = Mod_int<int32_t>,
			.int64_ = Mod_int64,
			.uint8_ = Mod_int<uint8_t>,
			.uint16_ = Mod_int<uint16_t>,
			.uint32_ = Mod_int<uint32_t>,
			.uint64_ = Mod_uint64,
			.bfloat16_ = Mod_bfloat16,
			.float16_ = Mod_float16,
			.float32_ = Mod_float32,
			.float64_ = Mod_float64,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 10) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Mod_int<int8_t>,
			.int16_ = Mod_int<int16_t>,
			.int32_ = Mod_int<int32_t>,
			.int64_ = Mod_int64,
			.uint8_ = Mod_int<uint8_t>,
			.uint16_ = Mod_int<uint16_t>,
			.uint32_ = Mod_int<uint32_t>,
			.uint64_ = Mod_uint64,
			.float16_ = Mod_float16,
			.float32_ = Mod_float32,
			.float64_ = Mod_float64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Mod_init;
		n->exit = Mod_exit;
		n->reshape = Mod_reshape;
	}
}
