#pragma once

#include <complex>
#include <memory>
#include "onnx.h"

#include "bool.h"
#include "float16.h"
#include "bfloat16.h"

namespace onnx {

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

template <int num> struct num_to_type;
template <> struct num_to_type<ONNX_TENSOR_TYPE_BOOL> { using T = bool_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_INT8> { using T = int8_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_INT16> { using T = int16_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_INT32> { using T = int32_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_INT64> { using T = int64_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_UINT8> { using T = uint8_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_UINT16> { using T = uint16_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_UINT32> { using T = uint32_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_UINT64> { using T = uint64_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_BFLOAT16> { using T = bfloat16_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_FLOAT16> { using T = float16_t; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_FLOAT32> { using T = float; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_FLOAT64> { using T = double; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_COMPLEX64> { using T = std::complex<float>; };
template <> struct num_to_type<ONNX_TENSOR_TYPE_COMPLEX128> { using T = std::complex<double>; };

template<typename T> struct type2int {};
template<> struct type2int<bool_t> { static constexpr int v = ONNX_TENSOR_TYPE_BOOL; };
template<> struct type2int<int8_t> { static constexpr int v = ONNX_TENSOR_TYPE_INT8; };
template<> struct type2int<int16_t> { static constexpr int v = ONNX_TENSOR_TYPE_INT16; };
template<> struct type2int<int32_t> { static constexpr int v = ONNX_TENSOR_TYPE_INT32; };
template<> struct type2int<int64_t> { static constexpr int v = ONNX_TENSOR_TYPE_INT64; };
template<> struct type2int<uint8_t> { static constexpr int v = ONNX_TENSOR_TYPE_UINT8; };
template<> struct type2int<uint16_t> { static constexpr int v = ONNX_TENSOR_TYPE_UINT16; };
template<> struct type2int<uint32_t> { static constexpr int v = ONNX_TENSOR_TYPE_UINT32; };
template<> struct type2int<uint64_t> { static constexpr int v = ONNX_TENSOR_TYPE_UINT64; };
template<> struct type2int<bfloat16_t> { static constexpr int v = ONNX_TENSOR_TYPE_BFLOAT16; };
template<> struct type2int<float16_t> { static constexpr int v = ONNX_TENSOR_TYPE_FLOAT16; };
template<> struct type2int<float> { static constexpr int v = ONNX_TENSOR_TYPE_FLOAT32; };
template<> struct type2int<double> { static constexpr int v = ONNX_TENSOR_TYPE_FLOAT64; };
template<> struct type2int<std::complex<float>> { static constexpr int v = ONNX_TENSOR_TYPE_COMPLEX64; };
template<> struct type2int<std::complex<double>> { static constexpr int v = ONNX_TENSOR_TYPE_COMPLEX128; };
template<> struct type2int<std::string> { static constexpr int v = ONNX_TENSOR_TYPE_STRING; };

template <typename T, typename ExecT>
__forceinline void exec_if(ExecT* self, tensor_type_t type) {
	if (type2int<T>::v == type) {
		self->template exec<T>();
	}
}

template <typename ExecT, typename ... Types>
__forceinline void typed_exec(ExecT* self, tensor_type_t type) {
	(exec_if<Types>(self, type), ...);
}

} // namespace onnx

