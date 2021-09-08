#pragma once

#include <onnx.h>

#define X(name) void resolver_default_op_ ## name(node_t* n);
namespace onnx {
#include "ops.h"
}
#undef X

extern onnx::resolver_t* resolver_default;
