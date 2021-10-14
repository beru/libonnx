#pragma once

#include <complex>
#include <memory>
#include "onnx.h"

#include "bool.h"
#include "float16.h"
#include "bfloat16.h"

namespace onnx {

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
inline void exec_if(ExecT* self, tensor_type_t type) {
	if (type2int<T>::v == type) {
		self->template exec<T>();
	}
}

template <typename ExecT, typename ... Types>
inline void typed_exec(ExecT* self, tensor_type_t type) {
	(exec_if<Types>(self, type), ...);
}

template <typename InputIt, typename T>
constexpr T multiply_accumulate( InputIt first, InputIt last, T init )
{
	for (; first != last; ++first) {
		init *= *first;
	}
	return init;
}

} // namespace onnx

