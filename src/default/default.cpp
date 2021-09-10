#include <default/default.h>

namespace onnx {

struct default_resolver : public resolver_t {

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

	void solve_operator(node_t* n) override {
		auto it = op_map.find(n->proto->op_type);
		if (it != op_map.end()) {
			it->second(n);
		}
	}

};

static default_resolver res;

extern resolver_t* resolver_default = &res;

} // namespace onnx

