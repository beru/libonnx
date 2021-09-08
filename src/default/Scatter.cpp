#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_Scatter(node_t* n)
{
	if (n->opset >= 11) {
		return;
	}else if (n->opset >= 9) {
	}
}

} // namespace onnx
