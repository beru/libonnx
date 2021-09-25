#include <default/default.h>

namespace onnx {

namespace {

struct default_resolver : public resolver_t {

	using ope_t = operator_t* (*)();
	std::map<std::string_view, ope_t> op_map;

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

	operator_t* solve_operator(std::string_view op_type) override {
		auto it = op_map.find(op_type);
		if (it != op_map.end()) {
			return it->second();
		}
		return nullptr;
	}

};

default_resolver res;

} // namespace {

extern resolver_t* resolver_default = &res;

} // namespace onnx

