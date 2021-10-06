#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct If_operator : public operator_t {
	std::unique_ptr<graph_t> else_branch;
	std::unique_ptr<graph_t> then_branch;

	bool init() override {
		if (!(inputs.size() == 1 && outputs.size() >= 1)) {
			return false;
		}
		else_branch.reset(new (std::nothrow) graph_t(ctx, attribute("else_branch", (Onnx__GraphProto*)nullptr)));
		then_branch.reset(new (std::nothrow) graph_t(ctx, attribute("then_branch", (Onnx__GraphProto*)nullptr)));
		if (!else_branch || !then_branch) {
			return false;
		}
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		const uint8_t* px = (const uint8_t*)x->data;
		graph_t* g;
		operator_t* t;

		if (px[0]) {
			g = then_branch.get();
		}else {
			g = else_branch.get();
		}
		if (g->nodes.size() > 0) {
			for (size_t i = 0; i < g->nodes.size(); i++) {
				t = g->nodes[i];
				t->reshape();
			}
			if (t) {
				for (size_t i = 0; i < min(t->outputs.size(), outputs.size()); i++) {
					tensor_t* a = t->outputs[i];
					tensor_t* b = outputs[i];
					b->reshape_identity(a);
				}
			}
		}
		return true;
	}

	void exec_impl() {
		const tensor_t* x = inputs[0];
		const uint8_t* px = (const uint8_t*)x->data;
		graph_t* g;
		operator_t* t;

		if (px[0]) {
			g = then_branch.get();
		}else {
			g = else_branch.get();
		}
		if (g->nodes.size() > 0) {
			for (size_t i = 0; i < g->nodes.size(); i++) {
				t = g->nodes[i];
				t->exec();
			}
			if (t) {
				for (size_t i = 0; i < min(t->outputs.size(), outputs.size()); i++) {
					tensor_t* a = t->outputs[i];
					tensor_t* b = outputs[i];
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

	void exec() override {
		if (opset >= 13) {
			exec_impl();
		}else if (opset >= 11) {
			exec_impl();
		}else if (opset >= 1) {
			exec_impl();
		}
	}
};

} // namespace {

operator_t* resolver_default_op_If(int opset)
{
	return new If_operator;
}

} // namespace onnx
