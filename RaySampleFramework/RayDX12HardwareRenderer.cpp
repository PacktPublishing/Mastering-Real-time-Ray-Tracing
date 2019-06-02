
#include "RayPCH.h"
#include "RayDX12HardwareRenderer.h"


using namespace Microsoft::WRL;


static inline void ThrowIfFailed(HRESULT hr)
{
	if (FAILED(hr))
	{
		throw std::exception();
	}
}

//Used to enable the debug layer 
void Ray_DX12HardwareRenderer::EnableDebugLayer()
{
#if defined(_DEBUG)
	// Always enable the debug layer before doing anything DX12 related
	// so all possible errors generated while creating DX12 objects
	// are caught by the debug layer.
	ComPtr<ID3D12Debug> debugInterface;
	ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
	debugInterface->EnableDebugLayer();
#endif
}


u64 Ray_DX12HardwareRenderer::Signal( u64& InFenceValue )
{
	u64 FenceValueForSignal = ++InFenceValue;
	ThrowIfFailed(mD3DCommandQueue->Signal(mFence.Get(), FenceValueForSignal));
	return FenceValueForSignal;
}


void Ray_DX12HardwareRenderer::WaitForFenceValue( u64 InFenceValue
					                           , HANDLE InFenceEvent
										       ,  std::chrono::milliseconds InDuration )
{
	if (mFence->GetCompletedValue() < InFenceValue)
	{
		ThrowIfFailed(mFence->SetEventOnCompletion(InFenceValue, InFenceEvent));
		::WaitForSingleObject(InFenceEvent, static_cast<DWORD>(InDuration.count()));
	}
}


void Ray_DX12HardwareRenderer::Flush( u64& InFenceValue
	                                , HANDLE InFenceEvent )
{
	u64 FenceValueForSignal = Signal(InFenceValue);
	WaitForFenceValue(FenceValueForSignal, InFenceEvent);

}


ComPtr<ID3D12Fence> Ray_DX12HardwareRenderer::CreateFence(ComPtr<ID3D12Device2> InDevice)
{
	ComPtr<ID3D12Fence> Fence;

	ThrowIfFailed(InDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence)));

	return Fence;
}


HANDLE Ray_DX12HardwareRenderer::CreateEventHandle()
{
	HANDLE FenceEvent;

	FenceEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	assert(FenceEvent && "Failed to create fence event.");

	return FenceEvent;
}


void Ray_DX12HardwareRenderer::Init(u32 InWidth, u32 InHeight,HWND InHwnd,bool InUseWarp)
{

	//Enable Debug Layer
	EnableDebugLayer();

	//Get the adapter
	mAdapter = GetAdapter(InUseWarp);

	//Create the device
	mD3DDevice = CreateDevice(mAdapter);

	//Let's create a command queue that can accept command lists of type DIRECT
	mD3DCommandQueue = CreateCommandQueue(mD3DDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);

	//Swap chain creation
	mSwapChain = CreateSwapChain(InHwnd, mD3DCommandQueue, InWidth, InHeight, kMAX_BACK_BUFFER_COUNT);

	//Get the index of the current swapchain backmbuffer
	mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

	//Create a descriptor heap of type RTV
	mRTVDescriptorHeap = CreateDescriptorHeap(mD3DDevice, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,kMAX_BACK_BUFFER_COUNT);

	//Get the size of the just created RTV heap 
	mRTVDescriptorSize = mD3DDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	//Update render target views
	UpdateRenderTargetViews(mD3DDevice, mSwapChain, mRTVDescriptorHeap);

	//Command list and command allocator creation
	for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
	{
		mD3DCommandAllocator[i] = CreateCommandAllocator(mD3DDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);
	}
	mD3DCommandList = CreateCommandList(mD3DDevice, mD3DCommandAllocator[mBackBufferIndex], D3D12_COMMAND_LIST_TYPE_DIRECT);


	//Create the dx12 fence
	mFence = CreateFence(mD3DDevice);
	
	//Create the CPU event that we'll use to stall the CPU on the fence value (the fence value will get signaled from the GPU as soon as the GPU will reach the fence)	
	mFenceEvent = CreateEventHandle();


}

void Ray_DX12HardwareRenderer::Destroy()
{

	// Make sure the command queue has finished all commands before closing.
	Flush(mFenceValue, mFenceEvent);

	//Close the CPU event 
	CloseHandle(mFenceEvent);

}

void Ray_DX12HardwareRenderer::Resize(u32 InWidth, u32 InHeight)
{
	if (mViewport.Width != InWidth || mViewport.Height != InHeight)
	{
		// Don't allow 0 size swap chain back buffers.
		u32 Width = std::max(1u, InWidth);
		u32 Height = std::max(1u, InHeight);
		
		mViewport.Width = static_cast<float>(Width);
		mViewport.Height = static_cast<float>(Height);

		// Flush the GPU queue to make sure the swap chain's back buffers
		// are not being referenced by an in-flight command list.
		Flush(mFenceValue, mFenceEvent);

		for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
		{
			// Any references to the back buffers must be released
			// before the swap chain can be resized.
			mRenderTargets[i].Reset();
			mFrameFenceValues[i] = mFrameFenceValues[mBackBufferIndex];
		}

		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		ThrowIfFailed(mSwapChain->GetDesc(&swapChainDesc));		

		ThrowIfFailed(mSwapChain->ResizeBuffers(kMAX_BACK_BUFFER_COUNT, Width, Height, swapChainDesc.BufferDesc.Format, swapChainDesc.Flags));

		mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

		UpdateRenderTargetViews(mD3DDevice, mSwapChain, mRTVDescriptorHeap);
	}
}



ComPtr<IDXGIAdapter4> Ray_DX12HardwareRenderer::GetAdapter(bool InUseWarp)
{
	ComPtr<IDXGIFactory4> dxgiFactory;
	UINT createFactoryFlags = 0;
#if defined(_DEBUG)
	createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));

	ComPtr<IDXGIAdapter1> dxgiAdapter1;
	ComPtr<IDXGIAdapter4> dxgiAdapter4;

	if (InUseWarp)
	{
		ThrowIfFailed(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter1)));
		ThrowIfFailed(dxgiAdapter1.As(&dxgiAdapter4));
	}
	else
	{
		SIZE_T maxDedicatedVideoMemory = 0;
		for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; ++i)
		{
			DXGI_ADAPTER_DESC1 dxgiAdapterDesc1;
			dxgiAdapter1->GetDesc1(&dxgiAdapterDesc1);

			// Check to see if the adapter can create a D3D12 device without actually 
			// creating it. The adapter with the largest dedicated video memory
			// is favored.
			if ((dxgiAdapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
				SUCCEEDED(D3D12CreateDevice(dxgiAdapter1.Get(),
					D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)) &&
				dxgiAdapterDesc1.DedicatedVideoMemory > maxDedicatedVideoMemory)
			{
				maxDedicatedVideoMemory = dxgiAdapterDesc1.DedicatedVideoMemory;
				ThrowIfFailed(dxgiAdapter1.As(&dxgiAdapter4));
			}
		}
	}

	return dxgiAdapter4;
}


ComPtr<ID3D12Device2> Ray_DX12HardwareRenderer::CreateDevice(ComPtr<IDXGIAdapter4> InAdapter)
{
	ComPtr<ID3D12Device2> d3d12Device2;

	ThrowIfFailed(D3D12CreateDevice(InAdapter.Get(),D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device2)));

	// Enable debug messages in debug mode.
#if defined(_DEBUG)
	ComPtr<ID3D12InfoQueue> pInfoQueue;
	if (SUCCEEDED(d3d12Device2.As(&pInfoQueue)))
	{
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

		// Suppress whole categories of messages
	 //D3D12_MESSAGE_CATEGORY Categories[] = {};

	 // Suppress messages based on their severity level
		D3D12_MESSAGE_SEVERITY Severities[] =
		{
			D3D12_MESSAGE_SEVERITY_INFO
		};

		// Suppress individual messages by their ID
		D3D12_MESSAGE_ID DenyIds[] = {
			D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,   // I'm really not sure how to avoid this message.
			D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                         // This warning occurs when using capture frame while graphics debugging.
			D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                       // This warning occurs when using capture frame while graphics debugging.
		};

		D3D12_INFO_QUEUE_FILTER NewFilter = {};
		//NewFilter.DenyList.NumCategories = _countof(Categories);
		//NewFilter.DenyList.pCategoryList = Categories;
		NewFilter.DenyList.NumSeverities = _countof(Severities);
		NewFilter.DenyList.pSeverityList = Severities;
		NewFilter.DenyList.NumIDs = _countof(DenyIds);
		NewFilter.DenyList.pIDList = DenyIds;

		ThrowIfFailed(pInfoQueue->PushStorageFilter(&NewFilter));
	}
#endif

	return d3d12Device2;
}


ComPtr<ID3D12CommandQueue> Ray_DX12HardwareRenderer::CreateCommandQueue(ComPtr<ID3D12Device2> InDevice, D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12CommandQueue> d3d12CommandQueue;

	D3D12_COMMAND_QUEUE_DESC desc = {};
	desc.Type = InType;
	desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	desc.NodeMask = 0;

	ThrowIfFailed(InDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&d3d12CommandQueue)));

	return d3d12CommandQueue;
}


/** Checks whether we support tearing or not */
static bool CheckTearingSupport()
{
	bool allowTearing = false;

	// Rather than create the DXGI 1.5 factory interface directly, we create the
	// DXGI 1.4 interface and query for the 1.5 interface. This is to enable the 
	// graphics debugging tools which will not support the 1.5 factory interface 
	// until a future update.
	ComPtr<IDXGIFactory4> factory4;
	if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))))
	{
		ComPtr<IDXGIFactory5> factory5;
		if (SUCCEEDED(factory4.As(&factory5)))
		{
			if (FAILED(factory5->CheckFeatureSupport(
				DXGI_FEATURE_PRESENT_ALLOW_TEARING, // <-- If we support this feature our adapter can support displays with variable refresh rate
				&allowTearing, sizeof(allowTearing))))
			{
				allowTearing = false;
			}
		}
	}

	return (allowTearing == true);
}

ComPtr<IDXGISwapChain4> Ray_DX12HardwareRenderer::CreateSwapChain(HWND InhWnd
	, ComPtr<ID3D12CommandQueue> InCommandQueue
	, u32 InWidth
	, u32 InHeight
	, u32 InBufferCount)
{
	ComPtr<IDXGISwapChain4> dxgiSwapChain4;
	ComPtr<IDXGIFactory4> dxgiFactory4;

	UINT CreateFactoryFlags = 0;

#if defined(_DEBUG)
	CreateFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(CreateDXGIFactory2(CreateFactoryFlags, IID_PPV_ARGS(&dxgiFactory4)));

	DXGI_SWAP_CHAIN_DESC1 SwapChainDesc = {};
	SwapChainDesc.Width = InWidth;
	SwapChainDesc.Height = InHeight;
	SwapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; //If we want to let the hw perform gamma correction for us we should use _SRGB format instead
	SwapChainDesc.Stereo = FALSE;
	SwapChainDesc.SampleDesc = { 1, 0 };
	SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	SwapChainDesc.BufferCount = InBufferCount;
	SwapChainDesc.Scaling = DXGI_SCALING_STRETCH;
	SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	SwapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

	// It is recommended to always allow tearing if tearing support is available.
	IsTearingSupported = CheckTearingSupport();
	SwapChainDesc.Flags = IsTearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	//Create Swapchain
	ComPtr<IDXGISwapChain1> swapChain1;
	ThrowIfFailed(dxgiFactory4->CreateSwapChainForHwnd(InCommandQueue.Get(),
		InhWnd,
		&SwapChainDesc,
		nullptr,
		nullptr,
		&swapChain1));

	// Disable the Alt+Enter fullscreen toggle feature. Switching to fullscreen
	// will be handled manually.
	ThrowIfFailed(dxgiFactory4->MakeWindowAssociation(InhWnd, DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain1.As(&dxgiSwapChain4));

	return dxgiSwapChain4;
}


ComPtr<ID3D12DescriptorHeap> Ray_DX12HardwareRenderer::CreateDescriptorHeap(ComPtr<ID3D12Device2> InDevice
	, D3D12_DESCRIPTOR_HEAP_TYPE InType
	, u32 InNumDescriptors)
{
	ComPtr<ID3D12DescriptorHeap> DescriptorHeap;

	D3D12_DESCRIPTOR_HEAP_DESC Desc = {};
	Desc.NumDescriptors = InNumDescriptors;
	Desc.Type = InType;

	ThrowIfFailed(InDevice->CreateDescriptorHeap(&Desc, IID_PPV_ARGS(&DescriptorHeap)));

	return DescriptorHeap;
}


void  Ray_DX12HardwareRenderer::UpdateRenderTargetViews(ComPtr<ID3D12Device2>        InDevice
	, ComPtr<IDXGISwapChain4>      InSwapChain
	, ComPtr<ID3D12DescriptorHeap> InDescriptorHeap)
{
	auto RTVDescriptorSize = InDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	CD3DX12_CPU_DESCRIPTOR_HANDLE RTVHandle(InDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
	{
		ComPtr<ID3D12Resource> backBuffer;
		ThrowIfFailed(InSwapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffer)));

		InDevice->CreateRenderTargetView(backBuffer.Get(), nullptr, RTVHandle);

		mRenderTargets[i] = backBuffer;

		RTVHandle.Offset(RTVDescriptorSize);
	}
}


ComPtr<ID3D12CommandAllocator> Ray_DX12HardwareRenderer::CreateCommandAllocator(ComPtr<ID3D12Device2>   InDevice
	, D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12CommandAllocator> CommandAllocator;
	ThrowIfFailed(InDevice->CreateCommandAllocator(InType, IID_PPV_ARGS(&CommandAllocator)));
	return CommandAllocator;
}



ComPtr<ID3D12GraphicsCommandList> Ray_DX12HardwareRenderer::CreateCommandList(ComPtr<ID3D12Device2> InDevice
	, ComPtr<ID3D12CommandAllocator> InCommandAllocator
	, D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12GraphicsCommandList> CommandList;

	ThrowIfFailed(InDevice->CreateCommandList(0, InType, InCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&CommandList)));

	//Before to reset a command list, we must close it.
	ThrowIfFailed(CommandList->Close());

	return CommandList;

}



void Ray_DX12HardwareRenderer::BeginFrame(float* InClearColor)
{

	//Beginning of the frame
	float DefaultClearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
	auto commandAllocator = mD3DCommandAllocator[mBackBufferIndex];
	auto backBuffer = mRenderTargets[mBackBufferIndex];

	//Before any commands can be recorded into the command list, the command allocator and command list needs to be reset to their initial state.
	commandAllocator->Reset();
	mD3DCommandList->Reset(commandAllocator.Get(), nullptr);

	//Before the render target can be cleared, it must be transitioned to the RENDER_TARGET state.

	// Clear the render target.
	{
		CD3DX12_RESOURCE_BARRIER Barrier = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

		mD3DCommandList->ResourceBarrier(1, &Barrier);

		//Now the back buffer can be cleared.
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(mRTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), mBackBufferIndex, mRTVDescriptorSize);

		mD3DCommandList->ClearRenderTargetView(rtv, InClearColor ? InClearColor : DefaultClearColor, 0, nullptr);
	}
}



void Ray_DX12HardwareRenderer::EndFrame()
{
	auto BackBuffer = mRenderTargets[mBackBufferIndex];

	CD3DX12_RESOURCE_BARRIER Barrier = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);

	mD3DCommandList->ResourceBarrier(1, &Barrier);

	//After transitioning to the correct state, the command list that contains the resource transition barrier must be executed on the command queue.
	ThrowIfFailed(mD3DCommandList->Close());

	ID3D12CommandList* const commandLists[] = { mD3DCommandList.Get() };

	mD3DCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

	u32 SyncInterval = mVSync ? 1 : 0;	
	u32 PresentFlags = IsTearingSupported && !mVSync ? DXGI_PRESENT_ALLOW_TEARING : 0;

	ThrowIfFailed(mSwapChain->Present(SyncInterval, PresentFlags));

	mFrameFenceValues[mBackBufferIndex] = Signal(mFenceValue);

	//After signaling the command queue, the index of the current back buffer is updated.
	mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

	//Before overwriting the contents of the current back buffer with the content of the next frame, the CPU thread is stalled using the WaitForFenceValue function described earlier.
	WaitForFenceValue(mFrameFenceValues[mBackBufferIndex], mFenceEvent);
}






