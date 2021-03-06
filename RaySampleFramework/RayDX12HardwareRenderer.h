#pragma once

#include "RayPCH.h"
#include "RayIHardwareRenderer.h"
#include "ShaderParameters.h"


#include <chrono>


struct RenderPacket
{
	RenderPacket() = default;

	// convenient name alias for verbose microsoft WRL ComPtr object
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	/** Vertex Buffer */
	ComPtr<ID3D12Resource> mVB;

	/** Index Buffer */
	ComPtr<ID3D12Resource> mIB;

	/** Vertex Count */
	u32 mVertexCount;

	/** Index Count */
	u32 mIndexCount;

};

struct RayTracingRenderPacket
{
	RayTracingRenderPacket() = default;

	// convenient name alias for verbose microsoft WRL ComPtr object
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	/** Bottom level acceleration structure */
	ComPtr<ID3D12Resource> mBLAS;

	/** Top level acceleration structure */
	ComPtr<ID3D12Resource> mTLAS;
};


class Ray_DX12HardwareRenderer : public Ray_IHardwareRenderer
{
public:

	Ray_DX12HardwareRenderer() = default;

	virtual void Init(u32 InWidth, u32 InHeight,HWND InHwnd,bool InUseWarp) override;

	virtual void Destroy() override;

	virtual void Resize(u32 InWidth, u32 InHeight) override;

	//NOTE: API  specific code must eventually go to the base class as we generalize more 

    /** Call this method before to begin a frame     */
	virtual void BeginFrame(float* InClearColor = nullptr) override;


	/** Call this method to render the actual frame */
	virtual void Render() override;


	/** Call this method at the end of a given frame perform the present*/
	virtual void EndFrame() override;


	// D3D12 specific methods
	/** Get the D3D command list */
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> GetCommandList() const { return mD3DCommandList; }

private:


	/** Wait for the previous frame to finish (this method will stall the CPU until the GPU finishes). 
	/** NOTE: This is NOT the proper way of synchronizing the CPU with the GPU */
	void WaitForPreviousFrame();

	/** Move to the next frame without blocking while the previous frame is in flight (only checks the previous frame has actually finished and eventually blocks. But in many cases the previous frame is done.) */
	/** This is the proper way of synchronizing the CPU with the GPU without stalling the CPU all the times */
	void MoveToNextFrame();

    // It is sometimes useful to wait until all previously executed commands have finished executing before doing something
    // (for example, resizing the swap chain buffers requires any references to the buffers to be released).
    // For this, the Flush function is used to ensure the GPU has finished processing all commands for each backbuffer before continuing. 

	/** Wait for the GPU to go idle */
	void FlushGPU();


	// convenient name alias for verbose microsoft WRL ComPtr object
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;


	/** Helper methods to create DX12 relevant objects */

	void EnableDebugLayer();

	
	ComPtr<IDXGIAdapter4> GetAdapter(bool InUseWarp = false);


	//ComPtr<ID3D12Device2> CreateDevice(ComPtr<IDXGIAdapter4> InAdapter);
	ComPtr<ID3D12Device5> CreateDevice(ComPtr<IDXGIAdapter4> InAdapter);

	
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



	//ComPtr<ID3D12GraphicsCommandList> CreateCommandList(  ComPtr<ID3D12Device2> InDevice
	//													, ComPtr<ID3D12CommandAllocator> InCommandAllocator
	//													, D3D12_COMMAND_LIST_TYPE InType);
	ComPtr<ID3D12GraphicsCommandList4> CreateCommandList(ComPtr<ID3D12Device2> InDevice
												      , ComPtr<ID3D12CommandAllocator> InCommandAllocator
		                                              , D3D12_COMMAND_LIST_TYPE InType);


	/** GPU synchronization relevant functions */

	ComPtr<ID3D12Fence> CreateFence(ComPtr<ID3D12Device2> InDevice);


	HANDLE CreateEventHandle();


	// Ray tracing specific methods ///////////////////////////////////////////////

	// TODO: Refactor thse methods. Some of them might be managed differently. Like heap descriptors might be created on demand and so on.
	// We want a system that can load ray tracing shaders and run on some kind of geometry 

	// Root signatures represent parameters passed to shaders 
	void CreateRootSignatures();

	// Ray tracing PSO creation (we eventually manage PSO with some kind of factory)
	void CreateRaytracingPipelineStateObject();

	// Create all the descriptor heap that we need here 
	void CreateDescriptorHeaps();

	// Build geometry 
	void BuildGeometry();

	// Build acceleration structures 
	void BuildAccelerationStructures();

	// Build shader tables, which define shaders and their local root arguments.
	void BuildShaderTables();


	// Create an output 2D texture to store the raytracing result to.
	void CreateRaytracingOutputResource();

	// Callback to recreate the output UAV if we resize the window
	void RecreateRaytracingOutputResource(u32 InWidth, u32 InHeight);

	// Used to copy the ray traced results to backbuffer in order to display them
	void CopyRayTracingOutputToBackBuffer();


	void SerializeAndCreateRaytracingRootSignature(D3D12_VERSIONED_ROOT_SIGNATURE_DESC& Desc, ComPtr<ID3D12RootSignature>* RootSig, D3D_ROOT_SIGNATURE_VERSION RootSignatureVersion);

	u32 AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* CPUDescriptor, u32 DescriptorIndexToUse);

	///////////////////////////////////////////////////////////////////////////////
	

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
	ComPtr<ID3D12Device5> mD3DDevice;                   //device for ray tracing

	/** D3D12 command queue */
	ComPtr<ID3D12CommandQueue> mD3DCommandQueue;

	/** D3D12 command list */	
	ComPtr<ID3D12GraphicsCommandList4> mD3DCommandList; //cmd list for ray tracing

	/** DirectX Ray tracing pipeline state object (PSO) */
	ComPtr<ID3D12StateObject> mDXRStateObject;

	/** D3D12 Command allocator */
	ComPtr<ID3D12CommandAllocator> mD3DCommandAllocators[kMAX_BACK_BUFFER_COUNT];



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
	
	/** Single fence used just to signal a given cmd list */
	ComPtr<ID3D12Fence> mFence;

	/** Fence values for each swap chain processed render target */
	u64 mFrameFenceValues[kMAX_BACK_BUFFER_COUNT] = { 0 };

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

	// Ray tracing specific structures and resources //////////////////////////////

	/** Used for TLAS, BLAS and Ray tracing output buffer*/
	ComPtr<ID3D12DescriptorHeap> mCBVSRVDescriptorHeap;
	u32 mDescriptorsNum;
	u32 mDescriptorSize;


	// Root signatures. Used to pass parameters to the shaders
	ComPtr<ID3D12RootSignature> mRaytracingGlobalRootSignature;
	ComPtr<ID3D12RootSignature> mRaytracingLocalRootSignature;

	// geometry data (vertex and index buffer)
	//ComPtr<ID3D12Resource> mVB;
	//ComPtr<ID3D12Resource> mIB;

	// Acceleration structure

	/** Bottom level acceleration structure */
	//ComPtr<ID3D12Resource> mBLAS;

	/** Top level acceleration structure */
	//ComPtr<ID3D12Resource> mTLAS;

	/** List of objects to be rendered (used by the rasterizer) */
	std::vector< RenderPacket > mRenderList;


	/** Ray tracing acceleration structure for a given scene (used by the ray tracing hardware) */
	RayTracingRenderPacket mRayTracingRenderPacket;


	/** Resource used to store ray tracing output */
	ComPtr<ID3D12Resource> mRayTracingOutputBuffer;
	D3D12_GPU_DESCRIPTOR_HANDLE mRaytracingOutputResourceUAVGpuDescriptor;
	UINT mRaytracingOutputResourceUAVDescriptorHeapIndex;

	 
	/** Keeps track of the allocated descriptors */
	u32 mAllocatedDescriptorsIndex = 0;


	/** This is the constand buffer we pass to the ray generation shader */
	RayGenCB mRayGenCB;

	// Shader tables
	ComPtr<ID3D12Resource> mMissShaderTable;
	ComPtr<ID3D12Resource> mHitGroupShaderTable;
	ComPtr<ID3D12Resource> mRayGenShaderTable;


	//////////////////////////////////////////////////////////////////////////////

    // Cbuffers

	/** Constant buffer resource  */
	ComPtr<ID3D12Resource> mSceneConstantsCB;

	/** Actual mapped heap for the constant buffer */
	SceneConstants* mSceneConstantsCB_DataPtr = nullptr;

	/** GPU side resource handle to the heap */
	CD3DX12_GPU_DESCRIPTOR_HANDLE mSceneConstantsHandle;

};
