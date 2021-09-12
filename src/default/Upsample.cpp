#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_Upsample()
{
	return nullptr;
	//if (opset >= 10) {
	//	return;
	//}else if (opset >= 9) {
	//}else if (opset >= 7) {
	//}
}

} // namespace onnx
