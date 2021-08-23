#include <onnx.h>
#include "util.h"

namespace {
} // namespace

void resolver_default_op_SequenceAt(onnx_node_t* n)
{
	if (n->opset >= 11) {
	}
}
