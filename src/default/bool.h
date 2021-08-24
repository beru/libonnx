#pragma once

struct bool_t final
{
	bool_t(){}
	bool_t(bool f) : val(f != false) {}
	operator bool() { return val != 0; }

	uint8_t val;
};
