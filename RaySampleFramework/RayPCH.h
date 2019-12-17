#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>

// C RunTime Header Files
#include <stdlib.h>
#include <sstream>
#include <iomanip>

#include <list>
#include <string>
#include <wrl.h>
#include <shellapi.h>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <atlbase.h>
#include <assert.h>

#include <dxgi1_6.h>
#include "d3d12_1.h"
#include <atlbase.h>
//#include "d3dx12.h"
// DirectX 12 specific headers.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#include "D3D12RaytracingHelpers.hpp"

// D3D12 extension library.
#include "d3dx12.h"


#ifdef _DEBUG
#include <dxgidebug.h>
#endif

#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif


//Basic multiplatform types redefinition
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;


// Useful macro for unused parameters
#define UNUSED_PARAMETER(x) ((void)(x))

// This is the base asset root dir
const std::string gAssetRootDir = "../Assets/";

// Helper to print on visual studio output 
template <typename ... Args>
void SmartPrintf(char const* const format,
	Args const& ... args) noexcept
{
#ifdef _DEBUG
	char Buf[128];
	sprintf_s(Buf, format, args ...);
	OutputDebugStringA(Buf);
#endif
}