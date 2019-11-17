#pragma once

//#include "RayPCH.h"

// Definition of viewport
struct Viewport
{
	float mLeft;
	float mTop;
	float mRight;
	float mBottom;
};

// Constant buffer that contains viewport and scissor rect
struct RayGenCB
{
	Viewport mViewport;
	Viewport mStencil;
};



struct SceneConstants
{
	DirectX::XMFLOAT4 CameraPosition;
};


// Handy way to the define global root parameters slot mapping
namespace GlobalRootSignatureParams
{
	enum Value
	{
		OutputViewSlot = 0,           // UAV slot
		AccelerationStructureSlot,    // Acceleration structure slot
		SceneConstantBuffer,          // Scene Constant buffer
		Count                         // Total number of global signature in use 
	};
}


// Handy way to the define local root parameters slot mapping
namespace LocalRootSignatureParams
{
	enum Value
	{
		ViewportConstantSlot = 0,    //CB slot (we pass viewport)
		Count
	};
}

