#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_GatherND()
{
	return nullptr;
	//if (opset >= 13) {
	//}else if (opset >= 12) {
	//}else if (opset >= 11) {
	//}
}

} // namespace onnx
