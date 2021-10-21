#pragma once

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <stdint.h>
#include <inttypes.h>

template <typename T> T min(T a, T b) { return a < b ? a : b; }
template <typename T> T max(T a, T b) { return a > b ? a : b; }
template <typename T> T clamp(T v, T a, T b) { return min(max(a, v), b); }

#include <stdio.h>
#ifdef __linux__
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <fenv.h>

/*
 * little or big endian
 */
#define ONNX_LITTLE_ENDIAN	(1)

static inline uint16_t __swab16(uint16_t x)
{
	return ((x << 8) | (x >> 8));
}

static inline uint32_t __swab32(uint32_t x)
{
	return ((x << 24) | (x >> 24) | \
		((x & (uint32_t)0x0000ff00UL) << 8) | \
		((x & (uint32_t)0x00ff0000UL) >> 8));
}

static inline uint64_t __swab64(uint64_t x)
{
	return ((x << 56) | (x >> 56) | \
		((x & (uint64_t)0x000000000000ff00ULL) << 40) | \
		((x & (uint64_t)0x0000000000ff0000ULL) << 24) | \
		((x & (uint64_t)0x00000000ff000000ULL) << 8) | \
		((x & (uint64_t)0x000000ff00000000ULL) >> 8) | \
		((x & (uint64_t)0x0000ff0000000000ULL) >> 24) | \
		((x & (uint64_t)0x00ff000000000000ULL) >> 40));
}

static inline uint32_t __swahw32(uint32_t x)
{
	return (((x & (uint32_t)0x0000ffffUL) << 16) | ((x & (uint32_t)0xffff0000UL) >> 16));
}

static inline uint32_t __swahb32(uint32_t x)
{
	return (((x & (uint32_t)0x00ff00ffUL) << 8) | ((x & (uint32_t)0xff00ff00UL) >> 8));
}

#ifdef ONNX_LITTLE_ENDIAN
#undef WORDS_BIGENDIAN
#else
#define WORDS_BIGENDIAN
#endif

#ifdef ONNX_LITTLE_ENDIAN
#define cpu_to_le64(x)		((uint64_t)(x))
#define le64_to_cpu(x)		((uint64_t)(x))
#define cpu_to_le32(x)		((uint32_t)(x))
#define le32_to_cpu(x)		((uint32_t)(x))
#define cpu_to_le16(x)		((uint16_t)(x))
#define le16_to_cpu(x)		((uint16_t)(x))
#define cpu_to_be64(x)		(__swab64((uint64_t)(x)))
#define be64_to_cpu(x)		(__swab64((uint64_t)(x)))
#define cpu_to_be32(x)		(__swab32((uint32_t)(x)))
#define be32_to_cpu(x)		(__swab32((uint32_t)(x)))
#define cpu_to_be16(x)		(__swab16((uint16_t)(x)))
#define be16_to_cpu(x)		(__swab16((uint16_t)(x)))
#else
#define cpu_to_le64(x)		(__swab64((uint64_t)(x)))
#define le64_to_cpu(x)		(__swab64((uint64_t)(x)))
#define cpu_to_le32(x)		(__swab32((uint32_t)(x)))
#define le32_to_cpu(x)		(__swab32((uint32_t)(x)))
#define cpu_to_le16(x)		(__swab16((uint16_t)(x)))
#define le16_to_cpu(x)		(__swab16((uint16_t)(x)))
#define cpu_to_be64(x)		((uint64_t)(x))
#define be64_to_cpu(x)		((uint64_t)(x))
#define cpu_to_be32(x)		((uint32_t)(x))
#define be32_to_cpu(x)		((uint32_t)(x))
#define cpu_to_be16(x)		((uint16_t)(x))
#define be16_to_cpu(x)		((uint16_t)(x))
#endif

/*
 * float16, bfloat16 and float32 conversion
 */
static inline uint16_t float32_to_float16(float v)
{
	union { uint32_t u; float f; } t;
	uint16_t y;

	t.f = v;
	y = ((t.u & 0x7fffffff) >> 13) - (0x38000000 >> 13);
	y |= ((t.u & 0x80000000) >> 16);
	return y;
}

static inline float float16_to_float32(uint16_t v)
{
	union { uint32_t u; float f; } t;

	t.u = v;
	t.u = ((t.u & 0x7fff) << 13) + 0x38000000;
	t.u |= ((v & 0x8000) << 16);
	return t.f;
}

static inline uint16_t float32_to_bfloat16(float v)
{
	union { uint32_t u; float f; } t;

	t.f = v;
	return t.u >> 16;
}

static inline float bfloat16_to_float32(uint16_t v)
{
	union { uint32_t u; float f; } t;

	t.u = v << 16;
	return t.f;
}

// https://handmade.network/forums/t/1507-compile_time_string_hashing_with_c++_constexpr_vs._your_own_preprocessor
constexpr uint32_t MK_FNV32_OFFSET_BASIS = 0x811c9dc5;
constexpr uint32_t MK_FNV32_PRIME = 16777619;
constexpr static uint32_t
_mk_const_hash_fnv1a32_null_terminated(const char* string)
{
	uint32_t hash = MK_FNV32_OFFSET_BASIS;
	while (*string) {
		hash = hash ^ (uint32_t)(*string++);
		hash = hash * MK_FNV32_PRIME;
	}
	return hash;
}
#define C_HASH(name) _mk_const_hash_fnv1a32_null_terminated( #name )
inline uint32_t c_hash(std::string_view str) { return _mk_const_hash_fnv1a32_null_terminated(str.data()); }

