#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_TopK()
{
	return nullptr;
	//if (opset >= 11) {
	//}else if (opset >= 10) {
	//}else if (opset >= 1) {
	//}
}

} // namespace onnx
