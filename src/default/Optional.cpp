#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
}

void resolver_default_op_Optional(node_t* n)
{
	if (n->opset >= 15) {
	}
}

} // namespace onnx
