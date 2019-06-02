#pragma once

#include "RayPCH.h"
#include "RayWin32Application.h"
#include "RayIHardwareRenderer.h"

//forward declaration to the renderer interface (virtualizes the underlying API)
//This will let us to specify even a different API other than DX12. Like Vulkan for example.
class Ray_IHardwareRenderer;

class Ray_Sample
{
public:

	Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName);

	virtual ~Ray_Sample();

	virtual void OnInit() = 0;

	virtual void OnUpdate(float DeltaFrame = 0.0f) = 0;

	virtual void OnRender() = 0;

	virtual void OnDestroy() = 0;

	void SetWidth(u32 InWidth) { mWidth = InWidth; }

	void SetHeight(u32 InHeight) { mHeight = InHeight; }

	u32 GetWidth() const { return mWidth; }
 
	u32 GetHeight() const { return mHeight;  }

	/** Is the hardware interface using a warp device ? */
	void SetUseWarp(bool InUseWarp) { mUseWarp = InUseWarp; }

	const wchar_t* GetSampleName() const { return mSampleName.c_str(); }

	void ResizeWindow(u32 ClientWidth, u32 ClientHeight);
		 
	Ray_IHardwareRenderer* GetHardwareRenderer() const { return mHardwareRenderer.get(); }

protected:

	/** Viewport relevant data members */
	u32 mWidth;
	
	u32 mHeight;

	float mAspectRatio = 0.0f;

	bool mUseWarp = false;

	/** A pointer to the hardware renderer. It virtualizes the underlying graphics API */	
	std::unique_ptr<Ray_IHardwareRenderer> mHardwareRenderer;	


private:

	/** The name of the sample that is shown on the window title */
	std::wstring mSampleName;

};
