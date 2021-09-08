#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

void resolver_default_op_NegativeLogLikelihoodLoss(node_t* n)
{
	if (n->opset >= 13) {
	}else if (n->opset >= 12) {
	}
}

} // namespace onnx
