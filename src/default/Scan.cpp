#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_Scan(node_t* n)
{
	if (n->opset >= 11) {
	}else if (n->opset >= 9) {
	}else if (n->opset >= 8) {
	}
}

} // namespace onnx
