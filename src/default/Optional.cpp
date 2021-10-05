#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
}

operator_t* resolver_default_op_Optional(int opset)
{
	//if (n->opset >= 15) {
	//}
	return nullptr;
}

} // namespace onnx
