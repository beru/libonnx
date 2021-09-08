#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct operator_pdata_t : public node_t::ope_pdata_t {
	~operator_pdata_t() {
		delete else_branch;
		delete then_branch;
	}

	graph_t * else_branch = nullptr;
	graph_t * then_branch = nullptr;
};

bool If_init(node_t* n)
{
	if (!(n->inputs.size() == 1 && n->outputs.size() >= 1)) {
		return false;
	}
	operator_pdata_t* pdat = new (std::nothrow) operator_pdata_t;
	if (!pdat)
		return false;
	pdat->else_branch = new (std::nothrow) graph_t(n->ctx, n->read_attribute("else_branch", (Onnx__GraphProto*)nullptr));
	pdat->then_branch = new (std::nothrow) graph_t(n->ctx, n->read_attribute("then_branch", (Onnx__GraphProto*)nullptr));
	if (!pdat->else_branch || !pdat->then_branch) {
		if (pdat->else_branch)
			delete pdat->else_branch;
		if (pdat->then_branch)
			delete pdat->then_branch;
		delete pdat;
		return false;
	}
	n->priv = pdat;
	return true;
}

int If_reshape(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	uint8_t* px = (uint8_t*)x->data;
	graph_t * g;
	node_t* t;
	int i;

	if (px[0])
		g = pdat->then_branch;
	else
		g = pdat->else_branch;
	if (g->nodes.size() > 0) {
		for (i = 0; i < g->nodes.size(); i++) {
			t = &g->nodes[i];
			t->reshape(t);
		}
		if (t) {
			for (i = 0; i < min(t->outputs.size(), n->outputs.size()); i++) {
				tensor_t* a = t->outputs[i];
				tensor_t* b = n->outputs[i];
				b->reshape_identity(a);
			}
		}
	}
	return 1;
}

void If_operator(node_t* n)
{
	operator_pdata_t* pdat = (operator_pdata_t*)n->priv;
	tensor_t* x = n->inputs[0];
	uint8_t* px = (uint8_t*)x->data;
	graph_t * g;
	node_t* t;
	int i;

	if (px[0])
		g = pdat->then_branch;
	else
		g = pdat->else_branch;
	if (g->nodes.size() > 0) {
		for (i = 0; i < g->nodes.size(); i++) {
			t = &g->nodes[i];
			t->ope(t);
		}
		if (t) {
			for (i = 0; i < min(t->outputs.size(), n->outputs.size()); i++) {
				tensor_t* a = t->outputs[i];
				tensor_t* b = n->outputs[i];
				if (x->type == ONNX_TENSOR_TYPE_STRING) {
					std::string* pa = (std::string*)a->data;
					std::string* pb = (std::string*)b->data;
					for (size_t o = 0; o < b->ndata; o++) {
						pb[o] = pa[o];
					}
				}else {
					memcpy(b->data, a->data, a->ndata * tensor_type_sizeof(a));
				}
			}
		}
	}
}

} // namespace

void resolver_default_op_If(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = If_operator;
	}else if (n->opset >= 11) {
		n->ope = If_operator;
	}else if (n->opset >= 1) {
		n->ope = If_operator;
	}
	if (n->ope) {
		n->init = If_init;
		n->reshape = If_reshape;
	}
}

} // namespace onnx
