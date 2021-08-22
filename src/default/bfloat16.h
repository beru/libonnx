#pragma once

#include <stdint.h>
#include "../onnxconf.h"

struct bfloat16_t
{
	bfloat16_t(){}
	bfloat16_t(float f) : val(float32_to_bfloat16(f)) {}
	operator float() { return bfloat16_to_float32(val); }

	bfloat16_t& operator += (bfloat16_t& v)
	{
		float f = *this;
		f += v;
		*this = f;
		return *this;
	}

	bfloat16_t& operator /= (bfloat16_t v)
	{
		float f = *this;
		f /= v;
		*this = f;
		return *this;
	}

	uint16_t val;
};

inline bfloat16_t operator + (bfloat16_t a, bfloat16_t b) { a += b; return a; }

inline bfloat16_t abs(bfloat16_t v) { return fabs(v); }
inline bfloat16_t acos(bfloat16_t v) { return acosf(v); }
inline bfloat16_t acosh(bfloat16_t v) { return acoshf(v); }
