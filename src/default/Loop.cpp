#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_Loop(node_t* n)
{
	if (n->opset >= 13) {
	}else if (n->opset >= 11) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
