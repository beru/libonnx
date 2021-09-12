#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_Scatter()
{
	return nullptr;
	//if (opset >= 11) {
	//	return;
	//}else if (opset >= 9) {
	//}
}

} // namespace onnx
