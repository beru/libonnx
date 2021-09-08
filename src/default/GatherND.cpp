#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_GatherND(node_t* n)
{
	if (n->opset >= 13) {
	}else if (n->opset >= 12) {
	}else if (n->opset >= 11) {
	}
}

} // namespace onnx
