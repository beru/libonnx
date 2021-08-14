#include <onnx.h>

struct ope_pdata_t {
	int isleft;
};

static int BitShift_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1)) {
		ope_pdata_t* pdat = new ope_pdata_t;
		pdat->isleft = (strcmp(n->attribute_read_string("direction", "LEFT"), "LEFT") == 0) ? 1 : 0;
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int BitShift_exit(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	delete pdat;
	return 1;
}

static int BitShift_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

static void BitShift_uint8(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	uint8_t* pa;
	uint8_t* pb;

	if (pdat->isleft) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint8_t*)a->broadcast_map_address(y, i);
			pb = (uint8_t*)b->broadcast_map_address(y, i);
			py[i] = *pa << *pb;
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint8_t*)a->broadcast_map_address(y, i);
			pb = (uint8_t*)b->broadcast_map_address(y, i);
			py[i] = *pa >> *pb;
		}
	}
}

static void BitShift_uint16(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint16_t* py = (uint16_t*)y->datas;
	uint16_t* pa;
	uint16_t* pb;

	if (pdat->isleft) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint16_t*)a->broadcast_map_address(y, i);
			pb = (uint16_t*)b->broadcast_map_address(y, i);
			py[i] = *pa << *pb;
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint16_t*)a->broadcast_map_address(y, i);
			pb = (uint16_t*)b->broadcast_map_address(y, i);
			py[i] = *pa >> *pb;
		}
	}
}

static void BitShift_uint32(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint32_t* py = (uint32_t*)y->datas;
	uint32_t* pa;
	uint32_t* pb;

	if (pdat->isleft) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint32_t*)a->broadcast_map_address(y, i);
			pb = (uint32_t*)b->broadcast_map_address(y, i);
			py[i] = *pa << *pb;
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint32_t*)a->broadcast_map_address(y, i);
			pb = (uint32_t*)b->broadcast_map_address(y, i);
			py[i] = *pa >> *pb;
		}
	}
}

static void BitShift_uint64(onnx_node_t* n)
{
	ope_pdata_t* pdat = (ope_pdata_t*)n->priv;
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint64_t* py = (uint64_t*)y->datas;
	uint64_t* pa;
	uint64_t* pb;

	if (pdat->isleft) {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint64_t*)a->broadcast_map_address(y, i);
			pb = (uint64_t*)b->broadcast_map_address(y, i);
			py[i] = *pa << *pb;
		}
	}else {
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			pa = (uint64_t*)a->broadcast_map_address(y, i);
			pb = (uint64_t*)b->broadcast_map_address(y, i);
			py[i] = *pa >> *pb;
		}
	}
}

void resolver_default_op_BitShift(onnx_node_t* n)
{
	if (n->opset >= 11) {
		n->ope = onnx_ope_type_selector{
			.uint8_ = BitShift_uint8,
			.uint16_ = BitShift_uint16,
			.uint32_ = BitShift_uint32,
			.uint64_ = BitShift_uint64,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = BitShift_init;
		n->exit = BitShift_exit;
		n->reshape = BitShift_reshape;
	}
}
