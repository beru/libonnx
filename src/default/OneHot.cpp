#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_OneHot(int opset)
{
	return nullptr;
	//if (opset >= 11) {
	//}else if (opset >= 9) {
	//}
}

} // namespace onnx
