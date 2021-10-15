#pragma once

#include "onnx.h"

namespace onnx {

operator_t* solve_operator(std::string_view op_type, int opset);

}

