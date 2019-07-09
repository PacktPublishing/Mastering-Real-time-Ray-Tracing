#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>


#define M_PI (3.1415927f)
#define ANSI_COLOR_RED     "\033..."
#define ANSI_COLOR_RESET   "\033..."

typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;


__host__ __device__ inline float CLAMP(float value, float low, float high)
{
	return min(high, max(value, low));
}

__device__ inline float RadToDeg(float Rad)
{
	return (Rad * 180.0f / M_PI);
}

__device__ inline float DegToRad(float Deg)
{
	return (Deg * M_PI / 180.0f);
}

// Useful random number generator
__host__ __device__ inline float GetRandom01(u32 *seed0, u32 *seed1)
{
	/* hash the seeds using bitwise AND operations and bitshifts */
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	u32 ires = ((*seed0) << 16) + (*seed1);

	/* use union struct to convert int to float */
	union {
		float f;
		u32 ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
}

template<typename T>
__device__ inline T Lerp(T a, T b, float t)
{
	return a + (b-a) * t;
}

