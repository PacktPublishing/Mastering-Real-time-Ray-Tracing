#pragma once

#include "RayPCH.h"
#include <chrono>


class Ray_IHardwareRenderer
{
public:

	virtual ~Ray_IHardwareRenderer() = default;

	virtual void Init(u32 InWidth, u32 InHeight,HWND InHwnd,bool InUseWarp = false) = 0;

	virtual void Destroy() = 0;

	virtual void Resize(u32 InWidth, u32 InHeight) = 0;

	//GPU Synchronization methods
	virtual u64 Signal(u64& InFenceValue) = 0;


	virtual void WaitForFenceValue(u64 InFenceValue
		                         , HANDLE InFenceEvent
		                         , std::chrono::milliseconds InDuration = std::chrono::milliseconds::max()) = 0;

	virtual void Flush(u64& InFenceValue
		             , HANDLE InFenceEvent) = 0;


	/** Call this method before to begin a frame     */
	virtual void BeginFrame(float* InClearColor = nullptr) = 0;

	/** Call this method to render the actual frame */
	virtual void Render() = 0;

	/** Wait for GPU to finish any pending work before to proceed*/
	virtual void WaitForGpuToFinish() = 0;

	/** Call this method at the end of a given frame perform the present*/
	virtual void EndFrame() = 0;

	/** Toggle VSync on/off */
	void ToggleVSync() {  mVSync = !mVSync;  }

protected:

	bool mVSync = false;


	/** Viewport relevant data members */
	u32 mWidth;

	u32 mHeight;

};
