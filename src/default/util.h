#pragma once

#include <complex>
#include <memory>
#include <onnx.h>

#include "bool.h"
#include "float16.h"
#include "bfloat16.h"

namespace onnx {

inline bool is_inout_size(const node_t* n, size_t in_size, size_t out_size) {
	return (n->inputs.size() == in_size) && (n->outputs.size() == out_size);
}

template <typename T, typename FuncT>
void foreach_tensor(node_t* n, FuncT func)
{
	const tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	const T* px = (const T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		py[i] = func(px[i]);
	}
}

using ope_t = void (*)(node_t* n);

template <typename T> constexpr bool is_type(tensor_type_t type);
template <> constexpr bool is_type<bool_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_BOOL; }
template <> constexpr bool is_type<int8_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT8; }
template <> constexpr bool is_type<int16_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT16; }
template <> constexpr bool is_type<int32_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT32; }
template <> constexpr bool is_type<int64_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT64; }
template <> constexpr bool is_type<uint8_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT8; }
template <> constexpr bool is_type<uint16_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT16; }
template <> constexpr bool is_type<uint32_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT32; }
template <> constexpr bool is_type<uint64_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT64; }
template <> constexpr bool is_type<bfloat16_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_BFLOAT16; }
template <> constexpr bool is_type<float16_t>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT16; }
template <> constexpr bool is_type<float>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT32; }
template <> constexpr bool is_type<double>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT64; }
template <> constexpr bool is_type<std::complex<float>>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_COMPLEX64; }
template <> constexpr bool is_type<std::complex<double>>(tensor_type_t type) { return type == ONNX_TENSOR_TYPE_COMPLEX128; }

template <template<typename> typename OperatorT, typename T, typename... Ts>
static std::shared_ptr<operator_t> ope_type_select(tensor_type_t type)
{
	if constexpr(sizeof...(Ts) == 0) {
		if (is_type<T>(type)) {
			return std::make_shared<OperatorT<T>>();
		}
	}else {
		return ope_type_select<OperatorT, Ts...>(type);
	}
}

} // namespace onnx

#define GEN_HOLEDR_TYPE(name, func) \
	struct name {\
		template <typename T> static void generic(node_t* n) {\
			func<T>(n);\
		}\
	};

