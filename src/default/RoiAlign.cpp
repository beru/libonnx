#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_RoiAlign(node_t* n)
{
	if (n->opset >= 10) {
	}
}

} // namespace onnx
