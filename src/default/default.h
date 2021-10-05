#pragma once

#include "onnx.h"

namespace onnx {
#define X(name) operator_t* resolver_default_op_ ## name(int opset);
#include "ops.h"
#undef X
extern resolver_t* resolver_default;
}

