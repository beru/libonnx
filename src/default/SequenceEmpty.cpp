#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {
} // namespace

operator_t* resolver_default_op_SequenceEmpty(int opset)
{
	return nullptr;
	//if (opset >= 11) {
	//}
}

} // namespace onnx
