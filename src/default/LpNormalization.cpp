#include <onnx.h>
#include "util.h"

namespace {
} // namespace

void resolver_default_op_LpNormalization(onnx_node_t* n)
{
	if (n->opset >= 1) {
	}
}
