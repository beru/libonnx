#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_CumSum(node_t* n)
{
	if (n->opset >= 14) {
	}else if (n->opset >= 11) {
	}
}

} // namespace onnx
