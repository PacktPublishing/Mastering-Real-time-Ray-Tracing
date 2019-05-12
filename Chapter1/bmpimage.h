#pragma once

#define _CRT_SECURE_NO_DEPRECATE
#include <stdint.h>

class Ray_BMPSaver
{

public:

	static void Save(const char* Filename,  int32_t Width,int32_t Height,int32_t dpi, float* Data);
};
