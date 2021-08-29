#include <default/default.h>

static int default_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	return y->reshape_identity(x);
}

struct default_resolver : public onnx_resolver_t {

	default_resolver() {
		name = "default";
		op_map = {
#define X(name) { #name, resolver_default_op_ ## name },
#include "ops.h"
#undef X
		};
	}

	void* create(void) override {
		return nullptr;
	}

	void destroy(void* rctx) override {
	}

	void solve_operator(onnx_node_t* n) override {
		auto it = op_map.find(n->proto->op_type);
		if (it != op_map.end()) {
			it->second(n);
			if (n->ope && !n->reshape) {
				n->reshape = default_reshape;
			}
		}
	}

};

static default_resolver res;

extern onnx_resolver_t* resolver_default = &res;

