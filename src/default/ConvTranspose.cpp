#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_ConvTranspose(node_t* n)
{
	if (n->opset >= 11) {
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
