#include <onnx.h>
#include "util.h"

namespace {
} // namespace

void resolver_default_op_SpaceToDepth(onnx_node_t* n)
{
	if (n->opset >= 13) {
	}else if (n->opset >= 1) {
	}
}
