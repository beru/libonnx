#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_QLinearConv(node_t* n)
{
	if (n->opset >= 10) {
	}
}

} // namespace onnx
