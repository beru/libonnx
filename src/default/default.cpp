#include <default/default.h>

void* resolver_default_create(void)
{
	return nullptr;
}

void resolver_default_destroy(void* rctx)
{
}

onnx_resolver_t resolver_default = {
	.name 							= "default",
	.create							= resolver_default_create,
	.destroy						= resolver_default_destroy,
	.op_map							= {
#define X(name) { #name, resolver_default_op_ ## name },
#include "ops.h"
#undef X
	},
};

