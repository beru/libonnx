#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct If_operator : public operator_t {
	std::unique_ptr<graph_t> else_branch;
	std::unique_ptr<graph_t> then_branch;

	bool init() override {
		if (!(n->inputs.size() == 1 && n->outputs.size() >= 1)) {
			return false;
		}
		else_branch.reset(new (std::nothrow) graph_t(n->ctx, n->attribute("else_branch", (Onnx__GraphProto*)nullptr)));
		then_branch.reset(new (std::nothrow) graph_t(n->ctx, n->attribute("then_branch", (Onnx__GraphProto*)nullptr)));
		if (!else_branch || !then_branch) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = n->inputs[0];
		const uint8_t* px = (const uint8_t*)x->data;
		graph_t * g;
		node_t* t;

		if (px[0])
			g = then_branch.get();
		else
			g = else_branch.get();
		if (g->nodes.size() > 0) {
			for (size_t i = 0; i < g->nodes.size(); i++) {
				t = &g->nodes[i];
				t->ope->reshape();
			}
			if (t) {
				for (size_t i = 0; i < min(t->outputs.size(), n->outputs.size()); i++) {
					tensor_t* a = t->outputs[i];
					tensor_t* b = n->outputs[i];
					b->reshape_identity(a);
				}
			}
		}
		return true;
	}

	void exec() override {
		const tensor_t* x = n->inputs[0];
		const uint8_t* px = (const uint8_t*)x->data;
		graph_t* g;
		node_t* t;

		if (px[0])
			g = then_branch.get();
		else
			g = else_branch.get();
		if (g->nodes.size() > 0) {
			for (size_t i = 0; i < g->nodes.size(); i++) {
				t = &g->nodes[i];
				t->ope->exec();
			}
			if (t) {
				for (size_t i = 0; i < min(t->outputs.size(), n->outputs.size()); i++) {
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
};

} // namespace {

void resolver_default_op_If(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = std::make_shared<If_operator>();
	}else if (n->opset >= 11) {
		n->ope = std::make_shared<If_operator>();
	}else if (n->opset >= 1) {
		n->ope = std::make_shared<If_operator>();
	}
}

} // namespace onnx
