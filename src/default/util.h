#pragma once

#include <complex>
#include <onnx.h>

#include "float16.h"
#include "bfloat16.h"

inline bool is_inout_size(onnx_node_t* n, size_t in_size, size_t out_size) {
	return (n->inputs.size() == in_size) && (n->outputs.size() == out_size);
}

using ope_t = void (*)(onnx_node_t* n);

struct onnx_ope_type_selector {
	ope_t bool_ = nullptr;
	ope_t int8_ = nullptr;
	ope_t int16_ = nullptr;
	ope_t int32_ = nullptr;
	ope_t int64_ = nullptr;
	ope_t uint8_ = nullptr;
	ope_t uint16_ = nullptr;
	ope_t uint32_ = nullptr;
	ope_t uint64_ = nullptr;
	ope_t bfloat16_ = nullptr;
	ope_t float16_ = nullptr;
	ope_t float32_ = nullptr;
	ope_t float64_ = nullptr;
	ope_t complex64_ = nullptr;
	ope_t complex128_ = nullptr;
	ope_t string_ = nullptr;

	ope_t select(onnx_tensor_type_t type) const {
		switch (type) {
		case ONNX_TENSOR_TYPE_INT8: return int8_;
		case ONNX_TENSOR_TYPE_INT16: return int16_;
		case ONNX_TENSOR_TYPE_INT32: return int32_;
		case ONNX_TENSOR_TYPE_INT64: return int64_;
		case ONNX_TENSOR_TYPE_UINT8: return uint8_;
		case ONNX_TENSOR_TYPE_UINT16: return uint16_;
		case ONNX_TENSOR_TYPE_UINT32: return uint32_;
		case ONNX_TENSOR_TYPE_UINT64: return uint64_;
		case ONNX_TENSOR_TYPE_BFLOAT16: return bfloat16_;
		case ONNX_TENSOR_TYPE_FLOAT16: return float16_;
		case ONNX_TENSOR_TYPE_FLOAT32: return float32_;
		case ONNX_TENSOR_TYPE_FLOAT64: return float64_;
		default: return nullptr;
		}
	}
};

using ope_t = void (*)(onnx_node_t* n);

template <typename T> constexpr bool is_type(onnx_tensor_type_t type);
template <> constexpr bool is_type<bool>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_BOOL; }
template <> constexpr bool is_type<int8_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT8; }
template <> constexpr bool is_type<int16_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT16; }
template <> constexpr bool is_type<int32_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT32; }
template <> constexpr bool is_type<int64_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_INT64; }
template <> constexpr bool is_type<uint8_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT8; }
template <> constexpr bool is_type<uint16_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT16; }
template <> constexpr bool is_type<uint32_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT32; }
template <> constexpr bool is_type<uint64_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_UINT64; }
template <> constexpr bool is_type<bfloat16_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_BFLOAT16; }
template <> constexpr bool is_type<float16_t>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT16; }
template <> constexpr bool is_type<float>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT32; }
template <> constexpr bool is_type<double>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_FLOAT64; }
template <> constexpr bool is_type<std::complex<float>>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_COMPLEX64; }
template <> constexpr bool is_type<std::complex<double>>(onnx_tensor_type_t type) { return type == ONNX_TENSOR_TYPE_COMPLEX128; }

template <typename HolderT, typename T, typename... Ts>
static ope_t onnx_ope_type_select(onnx_tensor_type_t type)
{
	if constexpr(sizeof...(Ts) == 0) {
		if (is_type<T>(type)) {
			return HolderT::template generic<T>;
		}else {
			return nullptr;
		}
	}else {
		return onnx_ope_type_select<HolderT, Ts...>(type);
	}
}

#define GEN_HOLEDR_TYPE(name, func) \
	struct name {\
		template <typename T> static void generic(onnx_node_t* n) {\
			func<T>(n);\
		}\
	};

