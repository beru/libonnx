#pragma once

#include <complex>
#include <memory>
#include <onnx.h>

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

#define CASE1(t1) case type2int<t1>::v: exec<t1>(); return;
#define CASE2(t1,...) CASE1(t1) CASE1(__VA_ARGS__)
#define CASE3(t1,...) CASE1(t1) CASE2(__VA_ARGS__)
#define CASE4(t1,...) CASE1(t1) CASE3(__VA_ARGS__)
#define CASE5(t1,...) CASE1(t1) CASE4(__VA_ARGS__)
#define CASE6(t1,...) CASE1(t1) CASE5(__VA_ARGS__)
#define CASE7(t1,...) CASE1(t1) CASE6(__VA_ARGS__)
#define CASE8(t1,...) CASE1(t1) CASE7(__VA_ARGS__)
#define CASE9(t1,...) CASE1(t1) CASE8(__VA_ARGS__)
#define CASE10(t1,...) CASE1(t1) CASE9(__VA_ARGS__)
#define CASE11(t1,...) CASE1(t1) CASE10(__VA_ARGS__)
#define CASE12(t1,...) CASE1(t1) CASE11(__VA_ARGS__)
#define CASE13(t1,...) CASE1(t1) CASE12(__VA_ARGS__)
#define CASE14(t1,...) CASE1(t1) CASE13(__VA_ARGS__)
#define CASE15(t1,...) CASE1(t1) CASE14(__VA_ARGS__)
#define CASE16(t1,...) CASE1(t1) CASE15(__VA_ARGS__)

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,NAME,...) NAME

#define TYPED_EXEC(type, ...) \
switch (type) { \
GET_MACRO(__VA_ARGS__, CASE16, CASE15, CASE14, CASE13, CASE12, CASE11, CASE10, CASE9, CASE8, CASE7, CASE6, CASE5, CASE4, CASE3, CASE2, CASE1)(__VA_ARGS__) \
}


} // namespace onnx

