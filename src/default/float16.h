#pragma once

#include <stdint.h>
#include "../onnxconf.h"

struct float16_t final
{
	float16_t(){}
	float16_t(float f) : val(float32_to_float16(f)) {}
	operator float() const { return float16_to_float32(val); }

	float16_t& operator += (float16_t v)
	{
		float f = *this;
		f += v;
		*this = f;
		return *this;
	}

	float16_t& operator /= (float16_t v)
	{
		float f = *this;
		f /= v;
		*this = f;
		return *this;
	}

	uint16_t val;
};

inline float16_t operator + (float16_t a, float16_t b) { a += b; return a; }


inline float16_t abs(float16_t v) { return fabs(v); }
inline float16_t acos(float16_t v) { return acosf(v); }
inline float16_t acosh(float16_t v) { return acoshf(v); }
inline float16_t pow(float16_t base, float16_t exponent) { return powf(base, exponent); }

