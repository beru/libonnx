#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
}

operator_t* resolver_default_op_OptionalGetElement()
{
	//if (n->opset >= 15) {
	//}
	return nullptr;
}

} // namespace onnx

