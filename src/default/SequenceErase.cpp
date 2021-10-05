#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_SequenceErase(int opset)
{
	return nullptr;
	//if (opset >= 11) {
	//}
}

} // namespace onnx
