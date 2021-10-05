#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_Trilu(int opset)
{
	return nullptr;
	//if (opset >= 14) {
	//}
}

} // namespace onnx
