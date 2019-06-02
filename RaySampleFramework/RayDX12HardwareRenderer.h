#pragma once

#include "RayPCH.h"
#include "RayIHardwareRenderer.h"

#include <chrono>

class Ray_DX12HardwareRenderer : public Ray_IHardwareRenderer
{
public:

	Ray_DX12HardwareRenderer() = default;

	virtual void Init(u32 InWidth, u32 InHeight,HWND InHwnd,bool InUseWarp) override;

	virtual void Destroy() override;

	virtual void Resize(u32 InWidth, u32 InHeight) override;


	virtual u64 Signal(u64& InFenceValue) override;


	virtual void WaitForFenceValue(u64 InFenceValue
		                 , HANDLE InFenceEvent
		                 , std::chrono::milliseconds InDuration = std::chrono::milliseconds::max()) override;


	
	// It is sometimes useful to wait until all previously executed commands have finished executing before doing something
	// (for example, resizing the swap chain buffers requires any references to the buffers to be released).
	// For this, the Flush function is used to ensure the GPU has finished processing all commands before continuing. 
    
	/** Flush the GPU  */
	virtual void Flush(u64& InFenceValue
		             , HANDLE InFenceEvent) override;


	//NOTE: API  specific code must eventually go to the base class as we generalize more 

    /** Call this method before to begin a frame     */
	virtual void BeginFrame(float* InClearColor = nullptr) override;


	/** Call this method at the end of a given frame perform the present*/
	virtual void EndFrame() override;

	/** Get the D3D command list */
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetCommandList() const { return mD3DCommandList; }

private:

	//convenient name alias for verbose microsoft WRL ComPtr object
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;


	/** Helper methods to create DX12 relevant objects */

	void EnableDebugLayer();

	
	ComPtr<IDXGIAdapter4> GetAdapter(bool InUseWarp = false);


	ComPtr<ID3D12Device2> CreateDevice(ComPtr<IDXGIAdapter4> InAdapter);

	
	ComPtr<ID3D12CommandQueue> CreateCommandQueue(ComPtr<ID3D12Device2> InDevice, D3D12_COMMAND_LIST_TYPE InType);

        
	ComPtr<IDXGISwapChain4> CreateSwapChain(HWND InhWnd
		                                  , ComPtr<ID3D12CommandQueue> InCommandQueue
		                                  , u32 InWidth
		                                  , u32 InHeight
		                                  , u32 InBufferCount );


	ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(ComPtr<ID3D12Device2> InDevice
		                                            , D3D12_DESCRIPTOR_HEAP_TYPE InType
		                                            , u32 InNumDescriptors);


	void UpdateRenderTargetViews( ComPtr<ID3D12Device2>        InDevice
								, ComPtr<IDXGISwapChain4>      InSwapChain
								, ComPtr<ID3D12DescriptorHeap> InDescriptorHeap);


	ComPtr<ID3D12CommandAllocator> CreateCommandAllocator(ComPtr<ID3D12Device2>   InDevice
	                                                    , D3D12_COMMAND_LIST_TYPE InType);



	ComPtr<ID3D12GraphicsCommandList> CreateCommandList(  ComPtr<ID3D12Device2> InDevice
														, ComPtr<ID3D12CommandAllocator> InCommandAllocator
														, D3D12_COMMAND_LIST_TYPE InType);


	/** GPU synchronization relevant functions */

	ComPtr<ID3D12Fence> CreateFence(ComPtr<ID3D12Device2> InDevice);


	HANDLE CreateEventHandle();
	

	//DX12 specific interfaces/////////////////////////////////

	static const size_t kMAX_BACK_BUFFER_COUNT = 3;

	/** Do we support tearing ? */
	bool IsTearingSupported = false;

	/** Index to the current back buffer */
	u32 mBackBufferIndex = 0;

	/** Interface to the installed graphics adapter */
	ComPtr<IDXGIAdapter4> mAdapter;

	/** A description of the graphics adapter       */
	std::wstring mAdapterDescription;



	/** D3D12 Device */
	ComPtr<ID3D12Device2> mD3DDevice;

	/** D3D12 command queue */
	ComPtr<ID3D12CommandQueue> mD3DCommandQueue;

	/** D3D12 command list */
	ComPtr<ID3D12GraphicsCommandList> mD3DCommandList;

	/** D3D12 Command allocator */
	ComPtr<ID3D12CommandAllocator> mD3DCommandAllocator[kMAX_BACK_BUFFER_COUNT];



	/** DXGI Factory used to create swap chain objects */
	ComPtr<IDXGIFactory4>               mDXGIFactory;

	/** Actua swap chain interface */
	ComPtr<IDXGISwapChain4>             mSwapChain;

	/** Render targets allocated to hold swap chains triple buffers */
	ComPtr<ID3D12Resource>              mRenderTargets[kMAX_BACK_BUFFER_COUNT];

	/** Depth stencil render target */
	ComPtr<ID3D12Resource>              mDepthStencil;


	// Synchronization objects

	/** Fence objects used to manage presentation */
	ComPtr<ID3D12Fence> mFence;

	/** Fence values for each swap chain processed render target */
	u64 mFrameFenceValues[kMAX_BACK_BUFFER_COUNT];

	/** Fence value used during synchronization */
	u64 mFenceValue = 0;

	/** Fence event */
	HANDLE mFenceEvent = nullptr;



	/** Heap descriptor used to allocate memory for swap-chain render targets */
	ComPtr<ID3D12DescriptorHeap> mRTVDescriptorHeap;

	/** Heap descriptor used to allocate memory for depth stencil render targets */
	ComPtr<ID3D12DescriptorHeap> mDSVDescriptorHeap;

	/** Heap descriptor size for render targets */
	u32 mRTVDescriptorSize;

	/** Screen viewport */
	D3D12_VIEWPORT mViewport;

	/** Scissor rect */
	D3D12_RECT  mScissorRect;

	/** The pixel format of the back buffer */
	DXGI_FORMAT mBackBufferFormat;

	/**  The pixel format of the depth buffer */
	DXGI_FORMAT mDepthBufferFormat;

	/** The number of back buffer held by the swap-chain */
	u32 mBackBufferCount;

	/** The minimum D3D feature level we must support  */
	D3D_FEATURE_LEVEL mD3DMinFeatureLevel;

	///////////////////////////////////////////////////////////

};
