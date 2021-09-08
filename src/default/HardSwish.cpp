#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_HardSwish(node_t* n)
{
	if (n->opset >= 14) {
	}
}

} // namespace onnx
