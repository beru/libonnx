#include <onnx.h>
#include "util.h"

namespace {
} // namespace

void resolver_default_op_MaxUnpool(onnx_node_t* n)
{
	if (n->opset >= 11) {
	}else if (n->opset >= 9) {
	}
}
