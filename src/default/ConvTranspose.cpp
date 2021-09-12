#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_ConvTranspose()
{
	return nullptr;
	//if (n->opset >= 11) {
	//}else if (n->opset >= 1) {
	//}
}

} // namespace onnx
