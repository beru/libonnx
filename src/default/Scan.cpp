#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_Scan()
{
	return nullptr;
	//if (opset >= 11) {
	//}else if (opset >= 9) {
	//}else if (opset >= 8) {
	//}
}

} // namespace onnx
