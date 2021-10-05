#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_Einsum(int opset)
{
	return nullptr;
	//if (opset >= 12) {
	//}
}

} // namespace onnx
