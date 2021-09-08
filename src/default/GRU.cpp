#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_GRU(node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 7) {
	}else if (n->opset >= 3) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
