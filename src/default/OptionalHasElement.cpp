#include <onnx.h>
#include "util.h"

namespace {
}

void resolver_default_op_OptionalHasElement(onnx_node_t* n)
{
	if (n->opset >= 15) {
	}
}

