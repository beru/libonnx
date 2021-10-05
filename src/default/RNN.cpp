#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_RNN(int opset)
{
	return nullptr;
	//if (opset >= 14) {
	//}else if (opset >= 7) {
	//}else if (opset >= 1) {
	//}
}

} // namespace onnx
