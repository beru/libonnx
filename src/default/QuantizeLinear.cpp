#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_QuantizeLinear()
{
	return nullptr;
	//if (opset >= 13) {
	//}else if (opset >= 10) {
	//}
}

} // namespace onnx
