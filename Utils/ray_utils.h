#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>


#define ANSI_COLOR_RED     "\033..."
#define ANSI_COLOR_RESET   "\033..."

typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

__constant__ const float PI = 3.1415927f;

__host__ __device__ inline float CLAMP(float value, float low, float high)
{
	return min(high, max(value, low));
}

__device__ inline float RadToDeg(float Rad)
{
	return (Rad * 180.0f / PI);
}

__device__ inline float DegToRad(float Deg)
{
	return (Deg * PI / 180.0f);
}



