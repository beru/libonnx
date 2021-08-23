#include <onnx.h>
#include "util.h"

namespace {
} // namespace

void resolver_default_op_NegativeLogLikelihoodLoss(onnx_node_t* n)
{
	if (n->opset >= 13) {
	}else if (n->opset >= 12) {
	}
}
