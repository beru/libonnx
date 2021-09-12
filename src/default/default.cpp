#include <default/default.h>

namespace onnx {

struct default_resolver : public resolver_t {

	using ope_t = operator_t* (*)();
	std::map<const char*, ope_t> op_map;

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

	operator_t* solve_operator(const char* op_type) override {
		auto it = op_map.find(op_type);
		if (it != op_map.end()) {
			return it->second();
		}
		return nullptr;
	}

};

static default_resolver res;

extern resolver_t* resolver_default = &res;

} // namespace onnx

