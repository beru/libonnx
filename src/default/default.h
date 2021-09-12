#pragma once

#include <onnx.h>

#define X(name) operator_t* resolver_default_op_ ## name();
namespace onnx {
#include "ops.h"
}
#undef X

extern onnx::resolver_t* resolver_default;
