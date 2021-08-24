#pragma once

#include <onnx.h>

void* resolver_default_create(void);
void resolver_default_destroy(void* rctx);

#define X(name) void resolver_default_op_ ## name(onnx_node_t* n);
#include "ops.h"
#undef X

extern onnx_resolver_t resolver_default;

