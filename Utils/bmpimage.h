#pragma once

#define _CRT_SECURE_NO_DEPRECATE
#include <stdint.h>

class Ray_BMP_Manager
{

public:

	static void Save(const char* Filename,  int32_t Width,int32_t Height,int32_t dpi, float* Data);

	static float* Load(const char* Filename, int32_t& Width, int32_t& Height);
};
