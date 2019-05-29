#pragma once

#include "RayPCH.h"
#include "RayWin32Application.h"

//forward declaration to the renderer interface (virtualizes the underlying API)
//This will let us to specify even a different API other than DX12. Like Vulkan for example.
//TODO: uncomment when DX12 will be moved to the hardware interface
//class Ray_IHardwareRenderer;

class Ray_Sample
{
public:

	Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName);

	virtual ~Ray_Sample();

	virtual void OnInit() = 0;

	virtual void OnUpdate(float DeltaFrame) = 0;

	virtual void OnRender() = 0;

	virtual void OnDestroy() = 0;

	u32 GetWidth() const { return mWidth; }
 
	u32 GetHeight() const { return mHeight;  }

	const wchar_t* GetSampleName() const { return mSampleName.c_str(); }

	void ResizeWindow(u32 ClientWidth, u32 ClientHeight);
		 
protected:

	/** Viewport relevant data members */
	u32 mWidth;
	
	u32 mHeight;

	float mAspectRatio = 0.0f;

	/** A pointer to the hardware renderer. It virtualizes the underlying graphics API */

	//TODO: Move the DX12 implementation in the hardware interface class. Do the same for any other API.
	//std::unique_ptr<Ray_IHardwareRenderer> mHardwareRenderer;	


private:

	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	//DX12 specific interfaces/////////////////////////////////

	static const size_t kMAX_BACK_BUFFER_COUNT = 3;

	/** Index to the current back buffer */
	u32 mBackBufferIndex = 0;

	/** Interface to the installed graphics adapter */
	ComPtr<IDXGIAdapter1> mAdapter;

	/** A description of the graphics adapter       */
	std::wstring mAdapterDescription;

	/** D3D12 Device */
	ComPtr<ID3D12Device> mD3DDevice;

	/** D3D12 command queue */
	ComPtr<ID3D12CommandQueue> mD3DCommandQueue;

	/** D3D12 command list */
	ComPtr<ID3D12CommandList> mD3DCommandList;
	
	/** D3D12 Command allocator */
	ComPtr<ID3D12CommandAllocator> mD3DCommandAllocator;




	///////////////////////////////////////////////////////////


	/** The name of the sample that is shown on the window title */
	std::wstring mSampleName;
};
