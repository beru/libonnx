#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_MaxRoiPool(node_t* n)
{
	if (n->opset >= 1) {
	}
}

} // namespace onnx
