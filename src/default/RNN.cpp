#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_RNN(node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 7) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
