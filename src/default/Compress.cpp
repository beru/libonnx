#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_Compress(node_t* n)
{
	if (n->opset >= 11) {
	}else if (n->opset >= 9) {
	}
}

} // namespace onnx
