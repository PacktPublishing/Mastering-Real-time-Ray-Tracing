
#include "RayPCH.h"
#include "RayDX12HardwareRenderer.h"

// Fbx loader singleton class 
#include "../Utils/fbxmodelloader.h"

// This header is produced after compilation
#include "RaytracingShaders.hlsl.h"


#define NAME_D3D12_OBJECT(x) ((x).Get()->SetName(L#x))
#define SizeOfInUint32(obj) ((sizeof(obj) - 1) / sizeof(u32) + 1)

using namespace Microsoft::WRL;

// Shader entry point names
static const wchar_t* gHitGroupName = L"RayCastingHitGroup";
static const wchar_t* gRaygenShaderName = L"RayCastingShader";
static const wchar_t* gClosestHitShaderName = L"RayCastingClosestHit";
static const wchar_t* gMissShaderName = L"RayCastingMiss";


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: Helpers that might be moved in a separated header file

//Handy helper functions
//NOTE: we could move those somewhere else (like in a specific header file for example)
static inline void ThrowIfFailed(HRESULT hr)
{
	if (FAILED(hr))
	{
		throw std::exception();
	}
}


static inline void ThrowIfFailed(HRESULT hr, const wchar_t* msg)
{
	if (FAILED(hr))
	{
		OutputDebugStringW(msg);
		throw std::exception();
	}
}


static inline void ThrowIfFalse(bool Value)
{
	ThrowIfFailed(Value ? S_OK : E_FAIL);
}

//Align power of two sizes
static inline u32 Align(u32 size, u32 alignment)
{
	return (size + (alignment - 1)) & ~(alignment - 1);
}


inline void AllocateUAVBuffer(ID3D12Device* Device, u64 BufferSize, ID3D12Resource **Resource, D3D12_RESOURCE_STATES InitialResourceState = D3D12_RESOURCE_STATE_COMMON, const wchar_t* ResourceName = nullptr)
{
	auto UploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto BufferDesc = CD3DX12_RESOURCE_DESC::Buffer(BufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	ThrowIfFailed(Device->CreateCommittedResource(
		&UploadHeapProperties,
		D3D12_HEAP_FLAG_NONE,
		&BufferDesc,
		InitialResourceState,
		nullptr,
		IID_PPV_ARGS(Resource)));
	if (ResourceName)
	{
		(*Resource)->SetName(ResourceName);
	}
}


inline void AllocateUploadBuffer(ID3D12Device* Device, void *Data, u64 Datasize, ID3D12Resource **Resource, const wchar_t* ResourceName = nullptr)
{
	auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(Datasize);
	
	ThrowIfFailed(Device->CreateCommittedResource(
		&uploadHeapProperties,
		D3D12_HEAP_FLAG_NONE,
		&bufferDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(Resource)));

	if (ResourceName)
	{
		(*Resource)->SetName(ResourceName);
	}

	// Upload resource
	void *pMappedData;
	(*Resource)->Map(0, nullptr, &pMappedData);
	memcpy(pMappedData, Data, Datasize);
	(*Resource)->Unmap(0, nullptr);
}


//Re-arrange these classes 

// This class is a wrapper around a resource that can be uploaded to the GPU from the CPU
class GpuUploadBuffer
{
public:
	ComPtr<ID3D12Resource> GetResource() { return m_resource; }

protected:
	ComPtr<ID3D12Resource> m_resource;

	GpuUploadBuffer() {}
	~GpuUploadBuffer()
	{
		if (m_resource.Get())
		{
			m_resource->Unmap(0, nullptr);
		}
	}

	void Allocate(ID3D12Device* device, UINT bufferSize, LPCWSTR resourceName = nullptr)
	{
		auto UploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

		auto BufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		ThrowIfFailed(device->CreateCommittedResource(
			&UploadHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&BufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_resource)));
		m_resource->SetName(resourceName);
	}

	u8* MapCpuWriteOnly()
	{
	    u8* MappedData;
		// We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
		CD3DX12_RANGE ReadRange(0, 0);        // We do not intend to read from this resource on the CPU.
		ThrowIfFailed(m_resource->Map(0, &ReadRange, reinterpret_cast<void**>(&MappedData)));
		return MappedData;
	}
};


// Shader record = {{Shader ID}, {RootArguments}}
class ShaderRecord
{
public:
	ShaderRecord(void* pShaderIdentifier, UINT shaderIdentifierSize) :
		shaderIdentifier(pShaderIdentifier, shaderIdentifierSize)
	{
	}

	ShaderRecord(void* pShaderIdentifier, UINT shaderIdentifierSize, void* pLocalRootArguments, UINT localRootArgumentsSize) :
		shaderIdentifier(pShaderIdentifier, shaderIdentifierSize),
		localRootArguments(pLocalRootArguments, localRootArgumentsSize)
	{
	}

	void CopyTo(void* dest) const
	{
		uint8_t* byteDest = static_cast<uint8_t*>(dest);
		memcpy(byteDest, shaderIdentifier.ptr, shaderIdentifier.size);
		if (localRootArguments.ptr)
		{
			memcpy(byteDest + shaderIdentifier.size, localRootArguments.ptr, localRootArguments.size);
		}
	}

	struct PointerWithSize {
		void *ptr;
		UINT size;

		PointerWithSize() : ptr(nullptr), size(0) {}
		PointerWithSize(void* _ptr, UINT _size) : ptr(_ptr), size(_size) {};
	};
	PointerWithSize shaderIdentifier;
	PointerWithSize localRootArguments;
};


// Shader table = {{ ShaderRecord 1}, {ShaderRecord 2}, ...}
class ShaderTable : public GpuUploadBuffer
{
	u8* m_mappedShaderRecords;
	UINT m_shaderRecordSize;

	// Debug support
	std::wstring m_name;
	std::vector<ShaderRecord> m_shaderRecords;

	ShaderTable() {}
public:
	ShaderTable(ID3D12Device* device, UINT numShaderRecords, UINT shaderRecordSize, LPCWSTR resourceName = nullptr)
		: m_name(resourceName)
	{
		m_shaderRecordSize = Align(shaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
		m_shaderRecords.reserve(numShaderRecords);
		UINT bufferSize = numShaderRecords * m_shaderRecordSize;
		Allocate(device, bufferSize, resourceName);
		m_mappedShaderRecords = MapCpuWriteOnly();
	}

	void push_back(const ShaderRecord& shaderRecord)
	{
		ThrowIfFalse(m_shaderRecords.size() < m_shaderRecords.capacity());
		m_shaderRecords.push_back(shaderRecord);
		shaderRecord.CopyTo(m_mappedShaderRecords);
		m_mappedShaderRecords += m_shaderRecordSize;
	}

	UINT GetShaderRecordSize() { return m_shaderRecordSize; }

	// Pretty-print the shader records.
	void DebugPrint(std::unordered_map<void*, std::wstring> shaderIdToStringMap)
	{
		std::wstringstream wstr;
		wstr << L"|--------------------------------------------------------------------\n";
		wstr << L"|Shader table - " << m_name.c_str() << L": "
			<< m_shaderRecordSize << L" | "
			<< m_shaderRecords.size() * m_shaderRecordSize << L" bytes\n";

		for (UINT i = 0; i < m_shaderRecords.size(); i++)
		{
			wstr << L"| [" << i << L"]: ";
			wstr << shaderIdToStringMap[m_shaderRecords[i].shaderIdentifier.ptr] << L", ";
			wstr << m_shaderRecords[i].shaderIdentifier.size << L" + " << m_shaderRecords[i].localRootArguments.size << L" bytes \n";
		}
		wstr << L"|--------------------------------------------------------------------\n";
		wstr << L"\n";
		OutputDebugStringW(wstr.str().c_str());
	}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//Used to enable the debug layer 
void Ray_DX12HardwareRenderer::EnableDebugLayer()
{
#if defined(_DEBUG)
	// Always enable the debug layer before doing anything DX12 related
	// so all possible errors generated while creating DX12 objects
	// are caught by the debug layer.
	ComPtr<ID3D12Debug> DebugInterface;
	ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&DebugInterface)));
	DebugInterface->EnableDebugLayer();
#endif
}




void Ray_DX12HardwareRenderer::WaitForPreviousFrame()
{
	// Signal and increment the fence value.
	const u64 Fence = mFenceValue;
	ThrowIfFailed(mD3DCommandQueue->Signal(mFence.Get(), Fence));
	mFenceValue++;

	// Wait until the previous frame is finished.
	if (mFence->GetCompletedValue() < Fence)
	{
		ThrowIfFailed(mFence->SetEventOnCompletion(Fence, mFenceEvent));
		WaitForSingleObject(mFenceEvent, INFINITE);
	}

	mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();
}


void Ray_DX12HardwareRenderer::MoveToNextFrame()
{
	// Schedule a Signal command in the queue.
	const u64 CurrentFenceValue = mFrameFenceValues[mBackBufferIndex];
	ThrowIfFailed(mD3DCommandQueue->Signal(mFence.Get(), CurrentFenceValue));

	// Update the frame index.
	mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (mFence->GetCompletedValue() < mFrameFenceValues[mBackBufferIndex])
	{
		ThrowIfFailed(mFence->SetEventOnCompletion(mFrameFenceValues[mBackBufferIndex], mFenceEvent));
		WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);
	}

	// Set the fence value for the next frame.
	mFrameFenceValues[mBackBufferIndex] = CurrentFenceValue + 1;

}


void Ray_DX12HardwareRenderer::FlushGPU()
{
	// Schedule a Signal command in the queue.
	ThrowIfFailed(mD3DCommandQueue->Signal(mFence.Get(), mFrameFenceValues[mBackBufferIndex]));

	// Wait until the fence has been processed.
	ThrowIfFailed(mFence->SetEventOnCompletion(mFrameFenceValues[mBackBufferIndex], mFenceEvent));
	WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);

	// Increment the fence value for the current frame.
	mFrameFenceValues[mBackBufferIndex]++;
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

// Helper method used by the CreateRootSignatures method to create global and local root signature
void Ray_DX12HardwareRenderer::SerializeAndCreateRaytracingRootSignature(D3D12_VERSIONED_ROOT_SIGNATURE_DESC& Desc, ComPtr<ID3D12RootSignature>* RootSig,D3D_ROOT_SIGNATURE_VERSION RootSignatureVersion)
{
	ComPtr<ID3DBlob> Blob;
	ComPtr<ID3DBlob> Error;

	ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&Desc, RootSignatureVersion, &Blob, &Error) , Error ? static_cast<wchar_t*>(Error->GetBufferPointer()) : nullptr);
	ThrowIfFailed(mD3DDevice->CreateRootSignature(1, Blob->GetBufferPointer(), Blob->GetBufferSize(), IID_PPV_ARGS(&(*RootSig))));
}

// Root signatures represent parameters passed to shaders 
void Ray_DX12HardwareRenderer::CreateRootSignatures()
{

	D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

	// This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
	featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

	if (FAILED(mD3DDevice->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
	{
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	// Global Root Signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	{
		CD3DX12_DESCRIPTOR_RANGE1 DescriptorRangeUAV;	
		CD3DX12_DESCRIPTOR_RANGE1 DescriptorRangeCBV;

		// We want to define a bounding convention for descriptor that fall in the range of UAV types.
		DescriptorRangeUAV.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
	
		
		CD3DX12_ROOT_PARAMETER1 RootParameters[GlobalRootSignatureParams::Count] = {};
		
		// UAV used to store ray tracing results (color buffer that will be used by the ray tracer to store the color)
		RootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &DescriptorRangeUAV);


		// Scene Constant buffer 
		DescriptorRangeCBV.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
		RootParameters[GlobalRootSignatureParams::SceneConstantBuffer].InitAsDescriptorTable(1, &DescriptorRangeCBV);

		//RootParameters[GlobalRootSignatureParams::SceneConstantBuffer].InitAsConstantBufferView(1, 0, D3D12_SHADER_VISIBILITY_ALL);				

		// Parameter mapping for the Top Level Acceleration structure passed for ray tracing 
		RootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
		
		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC GlobalRootSignatureDesc(ARRAYSIZE(RootParameters), RootParameters);		
		SerializeAndCreateRaytracingRootSignature(GlobalRootSignatureDesc, &mRaytracingGlobalRootSignature, featureData.HighestVersion);
		NAME_D3D12_OBJECT(mRaytracingGlobalRootSignature);
	}

	// Local Root Signature
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	{
		CD3DX12_ROOT_PARAMETER RootParameters[LocalRootSignatureParams::Count];


		RootParameters[LocalRootSignatureParams::ViewportConstantSlot].InitAsConstants(SizeOfInUint32(mRayGenCB), 0, 0);
		
		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC LocalRootSignatureDesc(ARRAYSIZE(RootParameters), RootParameters);
		
		// D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE - Denies the domain shader access to the root signature.
		// TODO: Check how this flag is related to ray tracing pipeline
		LocalRootSignatureDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		
		SerializeAndCreateRaytracingRootSignature(LocalRootSignatureDesc, &mRaytracingLocalRootSignature, featureData.HighestVersion);
		NAME_D3D12_OBJECT(mRaytracingLocalRootSignature);
	}

}



// Ray tracing PSO creation (we eventually manage PSO with some kind of factory)
void Ray_DX12HardwareRenderer::CreateRaytracingPipelineStateObject()
{
	CD3D12_STATE_OBJECT_DESC RaytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };


	// DXIL library
    // This contains the shaders and their entrypoints for the state object.
    // Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
	auto Lib = RaytracingPipeline.CreateSubobject<CD3D12_DXIL_LIBRARY_SUBOBJECT>();
	D3D12_SHADER_BYTECODE Libdxil = CD3DX12_SHADER_BYTECODE((void *)gRaytracingShaders, ARRAYSIZE(gRaytracingShaders)); // <- bytecode pointer 
	Lib->SetDXILLibrary(&Libdxil);

	// Define which shader exports to surface from the library.
    // If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
    // In this sample, this could be omitted for convenience since the sample uses all shaders in the library. 
	{
		Lib->DefineExport(gRaygenShaderName);
		Lib->DefineExport(gClosestHitShaderName);
		Lib->DefineExport(gMissShaderName);
	}

	// Triangle hit group
    // A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
    // In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
	auto HitGroup = RaytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
	HitGroup->SetClosestHitShaderImport(gClosestHitShaderName);
	HitGroup->SetHitGroupExport(gHitGroupName);
	HitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);


	// Shader config
    // Defines the maximum sizes in bytes for the ray payload and attribute structure.
	auto ShaderConfig = RaytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	u32 PayloadSize = 4 * sizeof(float);   // float4 color
	u32 AttributeSize = 2 * sizeof(float); // float2 barycentrics
	ShaderConfig->Config(PayloadSize, AttributeSize);


	// Local root signature and shader association
	// Hit group and miss shaders in this sample are not using a local root signature and thus one is not associated with them.
    {
		auto LocalRootSignatureSubObj = RaytracingPipeline.CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
		LocalRootSignatureSubObj->SetRootSignature(mRaytracingLocalRootSignature.Get());

		// Shader association
		auto RootSignatureAssociationSubObj = RaytracingPipeline.CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		RootSignatureAssociationSubObj->SetSubobjectToAssociate(*LocalRootSignatureSubObj);
		RootSignatureAssociationSubObj->AddExport(gRaygenShaderName);
	}

	// This is a root signature that enables a shader to have unique arguments that come from shader tables.

	// Global root signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	auto GlobalRootSignature = RaytracingPipeline.CreateSubobject<CD3D12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	GlobalRootSignature->SetRootSignature(mRaytracingGlobalRootSignature.Get());

	// Pipeline config
	// Defines the maximum TraceRay() recursion depth.
	auto PipelineConfigSubObj = RaytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	
	// PERFOMANCE TIP: Set max recursion depth as low as needed 
	// as drivers may apply optimization strategies for low recursion depths. 
	u32 MaxRecursionDepth = 1; // ~ primary rays only. 
	PipelineConfigSubObj->Config(MaxRecursionDepth);

	// Create a ray tracing pipeline state object (PSO) for the DXR pipeline
	ThrowIfFailed(mD3DDevice->CreateStateObject(RaytracingPipeline, IID_PPV_ARGS(&mDXRStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

// Create a heap for descriptors for ray tracing resource 
void Ray_DX12HardwareRenderer::CreateDescriptorHeaps()
{
	D3D12_DESCRIPTOR_HEAP_DESC DescriptorHeapDesc = {};
	
	// Create descriptor heap for CBV/SRV and UAV

	// Allocate a heap for 3 descriptors:
	// 1 - raytracing output texture UAV
	// 2 - bottom and top level acceleration structure fallback wrapped pointers
	// 1 - Constant buffer (scene constants)
	DescriptorHeapDesc.NumDescriptors = 2;
	DescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	DescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	DescriptorHeapDesc.NodeMask = 0;
	mD3DDevice->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(&mCBVSRVDescriptorHeap));
	NAME_D3D12_OBJECT(mCBVSRVDescriptorHeap);

	// Get the size of each descriptor on this hardware
	mDescriptorSize = mD3DDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create descriptor heap to store each RTV relative to each swapchain backbuffer (render target view)
	
	D3D12_DESCRIPTOR_HEAP_DESC Desc = {};
	Desc.NumDescriptors = kMAX_BACK_BUFFER_COUNT;
	Desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	ThrowIfFailed(mD3DDevice->CreateDescriptorHeap(&Desc, IID_PPV_ARGS(&mRTVDescriptorHeap)));

	// Get the size of each descriptor on this hardware
	mRTVDescriptorSize = mD3DDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV); 
}

// TODO: we build geometry in the hardware renderer for now, but we might move this elsewhere due to a refactor
// Build geometry 
void Ray_DX12HardwareRenderer::BuildGeometry()
{
	// Get the currnt d3d device
	auto Device = mD3DDevice.Get();

	// Loads vertices and indices from FBX file
	SModel model;
	FBXModelLoader::Get().Load((gAssetRootDir + "Cicada.fbx").c_str(), model);

	// This helper functions creates a buffer resource of type committed and upload vertex and index data in each one of them
	ComPtr<ID3D12Resource> VB;
	ComPtr<ID3D12Resource> IB;
	for (auto& Mesh : model.mMeshes)
	{
		// Allocate upload buffer for vertices 
		AllocateUploadBuffer(Device
			, &Mesh.mVertexBuffer[0]
			, sizeof(MyVertex)*Mesh.mVertexBuffer.size()
			, &VB);

		for (auto& MeshSection : Mesh.mMeshSections)
		{			
			if (MeshSection.mIndexBuffer)
			{

				// Allocate upload buffer for indices
				AllocateUploadBuffer(Device
					, MeshSection.mIndexBuffer
					, MeshSection.mIndexCount * (Mesh.mUse32BitIndices ? sizeof(u32) : sizeof(u16))
					, &IB);


				// NOTE: Renderer "connection" happens here. We actually create the GPU resource needed to render geometry ///////

				// Build Render Packet with vertex/index buffer geometry 
				RenderPacket RPacket;
				RPacket.mVB = VB;
				RPacket.mIB = IB;
				RPacket.mVertexCount = (u32)Mesh.mVertexBuffer.size();
				RPacket.mIndexCount = MeshSection.mIndexCount;
				mRenderList.push_back(RPacket);

				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			}
		}
	}
}

// Build acceleration structures 
void Ray_DX12HardwareRenderer::BuildAccelerationStructures()
{
	// Get current command allocator given the index of the current frame backbuffer
	auto CommnadAllocator = mD3DCommandAllocators[mBackBufferIndex].Get();
	auto Device = mD3DDevice.Get();

	// Reset the command list for the acceleration structure construction.
	mD3DCommandList->Reset(CommnadAllocator, nullptr);

	// Resource naming useful for debugging	
	std::wstring BLASName(L"BottomLevelAccelerationStructure_");
	std::wstring TLASName(L"TopLevelAccelerationStructure_");
	std::wstring InstanceDescName(L"InstanceDesc_");
	std::wstring ScratchResourceName(L"ScratchResource_");

	// Descs/Resources that need to be alive until they are potentially in use by the command list 
	ComPtr<ID3D12Resource> InstanceDescs;
	ComPtr<ID3D12Resource> ScratchResources;

	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> GeometryDescs;
	for (auto& RPacket : mRenderList)
	{		

		bool Use32BitIndices = (RPacket.mVertexCount > 65536);

		D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
		geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		geometryDesc.Triangles.IndexBuffer = RPacket.mIB->GetGPUVirtualAddress();
		geometryDesc.Triangles.IndexCount = RPacket.mIndexCount;
		geometryDesc.Triangles.IndexFormat = Use32BitIndices ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
		geometryDesc.Triangles.Transform3x4 = 0;
		geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
		geometryDesc.Triangles.VertexCount = RPacket.mVertexCount; 
		geometryDesc.Triangles.VertexBuffer.StartAddress = RPacket.mVB->GetGPUVirtualAddress();
		geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(MyVertex);

		// Mark the geometry as opaque. 
		// PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
		// Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
		geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

		// Store the geometry desc for this mesh
		GeometryDescs.push_back(geometryDesc);

	}

	// Get required sizes for an acceleration structure.
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS TopLevelInputs = {};
	TopLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	TopLevelInputs.Flags = buildFlags;
	TopLevelInputs.NumDescs = 1;
	TopLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO TopLevelPrebuildInfo = {};
	mD3DDevice->GetRaytracingAccelerationStructurePrebuildInfo(&TopLevelInputs, &TopLevelPrebuildInfo);

	ThrowIfFalse(TopLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO BottomLevelPrebuildInfo = {};
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS BottomLevelInputs = TopLevelInputs;
	BottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	BottomLevelInputs.pGeometryDescs = GeometryDescs.data();
	BottomLevelInputs.NumDescs = static_cast<u32>(GeometryDescs.size());
	mD3DDevice->GetRaytracingAccelerationStructurePrebuildInfo(&BottomLevelInputs, &BottomLevelPrebuildInfo);
	ThrowIfFalse(BottomLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);		

	AllocateUAVBuffer(Device, std::max(TopLevelPrebuildInfo.ScratchDataSizeInBytes, BottomLevelPrebuildInfo.ScratchDataSizeInBytes), &ScratchResources, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, ScratchResourceName.c_str());

	// Allocate resources for acceleration structures.
	// Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
	// Default heap is OK since the application doesn’t need CPU read/write access to them. 
	// The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
	// and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
	//  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
	//  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
	{
		D3D12_RESOURCE_STATES InitialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;		
		AllocateUAVBuffer(Device, BottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, &mRayTracingRenderPacket.mBLAS, InitialResourceState, BLASName.c_str());
	}


	// Bottom Level Acceleration Structure desc
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC BottomLevelBuildDesc = {};
	{
		BottomLevelBuildDesc.Inputs = BottomLevelInputs;
		BottomLevelBuildDesc.ScratchAccelerationStructureData = ScratchResources->GetGPUVirtualAddress();
		BottomLevelBuildDesc.DestAccelerationStructureData = mRayTracingRenderPacket.mBLAS->GetGPUVirtualAddress();
	}


	// TOP LEVEL ACCELERATION STRUCTURE CREATION CODE ////////////////////////////////////////////////////////////////////////////////////////////////

	// Top level acceleration structure creation code starts from here on 


	{
		D3D12_RESOURCE_STATES InitialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
		AllocateUAVBuffer(Device, TopLevelPrebuildInfo.ResultDataMaxSizeInBytes, &mRayTracingRenderPacket.mTLAS, InitialResourceState, TLASName.c_str());
	}

	// Ray tracing instance desc needs BLAS virtual GPU address		

	// NOTE: we should have as many instances as are the BLAS we've created
	{
		D3D12_RAYTRACING_INSTANCE_DESC InstanceDesc = {};
		InstanceDesc.Transform[0][0] = InstanceDesc.Transform[1][1] = InstanceDesc.Transform[2][2] = 1.0f;
		InstanceDesc.InstanceMask = 1;
		//InstanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
		InstanceDesc.AccelerationStructure = mRayTracingRenderPacket.mBLAS->GetGPUVirtualAddress();
		AllocateUploadBuffer(Device, &InstanceDesc, sizeof(InstanceDesc), &InstanceDescs, InstanceDescName.c_str());
	}

	// Top Level Acceleration Structure desc
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC TopLevelBuildDesc = {};
	{
		TopLevelInputs.InstanceDescs = InstanceDescs->GetGPUVirtualAddress();
		TopLevelBuildDesc.Inputs = TopLevelInputs;
		TopLevelBuildDesc.ScratchAccelerationStructureData = ScratchResources->GetGPUVirtualAddress();
		TopLevelBuildDesc.DestAccelerationStructureData = mRayTracingRenderPacket.mTLAS->GetGPUVirtualAddress();
	}


	auto BuildAccelerationStructure = [&](auto* RaytracingCommandList)
	{
		RaytracingCommandList->BuildRaytracingAccelerationStructure(&BottomLevelBuildDesc, 0, nullptr);
		RaytracingCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(mRayTracingRenderPacket.mBLAS.Get()));
		RaytracingCommandList->BuildRaytracingAccelerationStructure(&TopLevelBuildDesc, 0, nullptr);
		RaytracingCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(mRayTracingRenderPacket.mTLAS.Get()));	
	};

	// Build acceleration structure.
	BuildAccelerationStructure(mD3DCommandList.Get());


	// Close the command list and Kick off acceleration structure construction for each object, by executing the recorded commands on the cmd queue on the GPU
	ThrowIfFailed(mD3DCommandList->Close());
	ID3D12CommandList *commandLists[] = { mD3DCommandList.Get() };
	mD3DCommandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

	// Create temporary fence and let's wait for the work in flight on the GPU cmd queue to complete

	// Actually create the fence with an initial value equal to 0
	ComPtr<ID3D12Fence> Fence;
	ThrowIfFailed(mD3DDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence)));
	
	// Fance value to be reached
	u64 LocalFenceValue = 1;

	// Signal the queue with the value to be reached
	mD3DCommandQueue->Signal(Fence.Get(), LocalFenceValue);

	// Check if the completed fence value has been reached
	if (Fence->GetCompletedValue() < LocalFenceValue)
	{
		// OS Event handler to be signaled once the GPU cmd queue reaches the fence
		Fence->SetEventOnCompletion(LocalFenceValue, mFenceEvent);

		// Block on the event until we reach LocalFenceValue
		WaitForSingleObject(mFenceEvent, INFINITE);
	}
}

// Build shader tables, which define shaders and their local root arguments.
void Ray_DX12HardwareRenderer::BuildShaderTables()
{
	auto device = mD3DDevice.Get();

	void* RayGenShaderIdentifier;
	void* MissShaderIdentifier;
	void* HitGroupShaderIdentifier;

	auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
	{
		RayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(gRaygenShaderName);
		MissShaderIdentifier = stateObjectProperties->GetShaderIdentifier(gMissShaderName);
		HitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier(gHitGroupName);
	};

	// Get shader identifiers.
	u32 shaderIdentifierSize;

	ComPtr<ID3D12StateObjectPropertiesPrototype> stateObjectProperties;
	ThrowIfFailed(mDXRStateObject.As(&stateObjectProperties));
	GetShaderIdentifiers(stateObjectProperties.Get());
	shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	

	// Ray gen shader table
	{
		struct RootArguments 
		{
			RayGenCB cb;
		} rootArguments;

		rootArguments.cb = mRayGenCB;

		u32 numShaderRecords = 1;
		u32 shaderRecordSize = shaderIdentifierSize + sizeof(rootArguments);
		ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
		rayGenShaderTable.push_back(ShaderRecord(RayGenShaderIdentifier, shaderIdentifierSize, &rootArguments, sizeof(rootArguments)));
		mRayGenShaderTable = rayGenShaderTable.GetResource();
	}

	// Miss shader table
	{
		u32 numShaderRecords = 1;
		u32 shaderRecordSize = shaderIdentifierSize;
		ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
		missShaderTable.push_back(ShaderRecord(MissShaderIdentifier, shaderIdentifierSize));
		mMissShaderTable = missShaderTable.GetResource();
	}

	// Hit group shader table
	{
		u32 numShaderRecords = 1;
		u32 shaderRecordSize = shaderIdentifierSize;
		ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");
		hitGroupShaderTable.push_back(ShaderRecord(HitGroupShaderIdentifier, shaderIdentifierSize));
		mHitGroupShaderTable = hitGroupShaderTable.GetResource();
	}

}


// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
u32 Ray_DX12HardwareRenderer::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* CPUDescriptor, u32 DescriptorIndexToUse)
{
	auto DescriptorHeapCpuBase = mCBVSRVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
	if (DescriptorIndexToUse >= mCBVSRVDescriptorHeap->GetDesc().NumDescriptors)
	{
		DescriptorIndexToUse = mAllocatedDescriptorsIndex++;
	}
	*CPUDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(DescriptorHeapCpuBase, DescriptorIndexToUse, mDescriptorSize);
	return DescriptorIndexToUse;
}


// Create an output 2D texture to store the raytracing result to.
void Ray_DX12HardwareRenderer::CreateRaytracingOutputResource()
{	
	// Create the output resource. The dimensions and format should match the swap-chain.

	// Unordered Access View Desc creation
	auto UAVResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(mBackBufferFormat, mWidth, mHeight, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	// Property flag for the heap descriptor that will contain this resource
	auto DefaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

	// ID3D12Resource creation, this resource will be an unordered access view
	ThrowIfFailed(mD3DDevice->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &UAVResourceDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mRayTracingOutputBuffer)));
	NAME_D3D12_OBJECT(mRayTracingOutputBuffer);

	// Allocate a descriptor and return a valid handle to it and a heap index for correct access into the decriptor table 
	D3D12_CPU_DESCRIPTOR_HANDLE UAVDescriptorHandle;	
	mRaytracingOutputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&UAVDescriptorHandle, mRaytracingOutputResourceUAVDescriptorHeapIndex);

	// Unordered access view desc
	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
	
	// Set the view dimension
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	mD3DDevice->CreateUnorderedAccessView(mRayTracingOutputBuffer.Get(), nullptr, &UAVDesc, UAVDescriptorHandle);
	
	mRaytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCBVSRVDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), mRaytracingOutputResourceUAVDescriptorHeapIndex, mDescriptorSize);
}


// Callback to recreate the output UAV if we resize the window
void Ray_DX12HardwareRenderer::RecreateRaytracingOutputResource(u32 InWidth, u32 InHeight)
{
	// Create the output resource. The dimensions and format should match the swap-chain.

	// Unordered Access View Desc creation
	auto UAVResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(mBackBufferFormat, InWidth, InHeight, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	// Property flag for the heap descriptor that will contain this resource
	auto DefaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

	// ID3D12Resource creation, this resource will be an unordered access view
	ThrowIfFailed(mD3DDevice->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &UAVResourceDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mRayTracingOutputBuffer)));
	NAME_D3D12_OBJECT(mRayTracingOutputBuffer);

	// Allocate a descriptor and return a valid handle to it and a heap index for correct access into the decriptor table 
	D3D12_CPU_DESCRIPTOR_HANDLE UAVDescriptorHandle;
	mRaytracingOutputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&UAVDescriptorHandle, mRaytracingOutputResourceUAVDescriptorHeapIndex);

	// Unordered access view desc
	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};

	// Set the view dimension
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	mD3DDevice->CreateUnorderedAccessView(mRayTracingOutputBuffer.Get(), nullptr, &UAVDesc, UAVDescriptorHandle);


	mRaytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCBVSRVDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), mRaytracingOutputResourceUAVDescriptorHeapIndex, mDescriptorSize);
}



void Ray_DX12HardwareRenderer::Init(u32 InWidth, u32 InHeight,HWND InHwnd,bool InUseWarp)
{
	// Let's cache width and height of the buffer
	mWidth  = InWidth;
	mHeight = InHeight;	

	//Enable Debug Layer
	EnableDebugLayer();

	//Get the adapter
	mAdapter = GetAdapter(InUseWarp);

	//Create the device
	mD3DDevice = CreateDevice(mAdapter);

	//Let's create a command queue that can accept command lists of type DIRECT
	mD3DCommandQueue = CreateCommandQueue(mD3DDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);
	NAME_D3D12_OBJECT(mD3DCommandQueue);	

	//Swap chain creation
	mSwapChain = CreateSwapChain(InHwnd, mD3DCommandQueue, InWidth, InHeight, kMAX_BACK_BUFFER_COUNT);


	// Create descriptor heaps used by this hardware renderer
	CreateDescriptorHeaps();

	
	// Create the actual render target view into the descriptor heap we've just created
	CD3DX12_CPU_DESCRIPTOR_HANDLE RTVHandle(mRTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
	for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
	{
		ComPtr<ID3D12Resource> backBuffer;
		ThrowIfFailed(mSwapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffer)));

		mD3DDevice->CreateRenderTargetView(backBuffer.Get(), nullptr, RTVHandle);

		mRenderTargets[i] = backBuffer;

		RTVHandle.Offset(mRTVDescriptorSize);
	}

	// Command list and command allocator creation

	// We need to create as many command allocator as are the buffer that we want the GPU to submit work for
	for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
	{
		mD3DCommandAllocators[i] = CreateCommandAllocator(mD3DDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);
	}

	// Create a command list for this application (i.e. application is single threaded therefore a single command list is enough in this case)
	mD3DCommandList = CreateCommandList(mD3DDevice, mD3DCommandAllocators[mBackBufferIndex], D3D12_COMMAND_LIST_TYPE_DIRECT);




	// Dynamic constant buffer creation here

    // Create the constant buffers.
	const u32 CBufferSize = (sizeof(SceneConstants) + (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1); // must be a multiple 256 bytes
	ThrowIfFailed( mD3DDevice->CreateCommittedResource( &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
														D3D12_HEAP_FLAG_NONE,
														&CD3DX12_RESOURCE_DESC::Buffer(CBufferSize),
														D3D12_RESOURCE_STATE_GENERIC_READ,
														nullptr,
														IID_PPV_ARGS(&mSceneConstantsCB)) );
	

	// Map the constant buffers and cache their heap pointers.
	CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
	ThrowIfFailed(mSceneConstantsCB->Map(0, &readRange, reinterpret_cast<void**>(&mSceneConstantsCB_DataPtr)));
	
	// Create the constant buffer views (for now just the one related to scene constants
	D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
	CBVDesc.SizeInBytes = CBufferSize;
	CBVDesc.BufferLocation = mSceneConstantsCB->GetGPUVirtualAddress();

	// Now we need to know where to put the constant buffer, therefore we provide for the correct heap handle
	// This handle is a CPU side handle that we use like a pointer (it is not a pointer obviously, but a safest way to reference a resource preventing from erroneous dereferencing)
	auto DescriptorHeapCpuBase = mCBVSRVDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
	auto CBufferCPUHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(DescriptorHeapCpuBase);
	CBufferCPUHandle.Offset(mDescriptorSize);

	// We cache the GPU handle which will be the one we will use to set the constat buffer
	mD3DDevice->CreateConstantBufferView(&CBVDesc, CBufferCPUHandle);
	mSceneConstantsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCBVSRVDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
	mSceneConstantsHandle.Offset(mDescriptorSize);


	// One fence is enough to track that the previous in flight work for the previous frame is done.
	// Create the dx12 fence
	mFence = CreateFence(mD3DDevice);
	++mFrameFenceValues[mBackBufferIndex];

	//Create the CPU event that we'll use to stall the CPU on the fence value (the fence value will get signaled from the GPU as soon as the GPU will reach the fence)	
	mFenceEvent = CreateEventHandle();
	

	// TODO: A heavy refactoring is needed for the ray tracing object creation and management part.
	// In particular we expect to manage heap descriptors, PSO etc. in separated systems/classes. This should promote more modularity and let the code to be more maintainable.

	// Ray Tracing objects init

    // Create root signatures for the shaders.
	CreateRootSignatures();

	// Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
	CreateRaytracingPipelineStateObject();

	// Build geometry to be used in the sample.
	BuildGeometry();

	// Build raytracing acceleration structures from the generated geometry.
	BuildAccelerationStructures();

	// Build shader tables, which define shaders and their local root arguments.
	BuildShaderTables();

	// Create an output 2D texture to store the raytracing result to.
	CreateRaytracingOutputResource();
}

void Ray_DX12HardwareRenderer::Destroy()
{

	// Make sure the command queue has finished all commands before closing.
	FlushGPU();

	//Close the CPU event 
	CloseHandle(mFenceEvent);
}

void Ray_DX12HardwareRenderer::Resize(u32 InWidth, u32 InHeight)
{
	if (mViewport.Width != InWidth || mViewport.Height != InHeight)
	{
		// Don't allow 0 size swap chain back buffers.
		const u32 Width = std::max(1u, InWidth);
		const u32 Height = std::max(1u, InHeight);
		
		mViewport.Width = static_cast<float>(Width);
		mViewport.Height = static_cast<float>(Height);

		mWidth = Width;
		mHeight = Height;

		// Flush the GPU queue to make sure the swap chain's back buffers
		// are not being referenced by an in-flight command list.
		FlushGPU();
		
		// Ready to release any reference to the back buffers
		for (int i = 0; i < kMAX_BACK_BUFFER_COUNT; ++i)
		{
			// Any references to the back buffers must be released
			// before the swap chain can be resized.
			mRenderTargets[i].Reset();			 
		}

		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		ThrowIfFailed(mSwapChain->GetDesc(&swapChainDesc));		

		ThrowIfFailed(mSwapChain->ResizeBuffers(kMAX_BACK_BUFFER_COUNT, Width, Height, swapChainDesc.BufferDesc.Format, swapChainDesc.Flags));

		mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

		// Update the render target views to match the new resized viewport 
		UpdateRenderTargetViews(mD3DDevice, mSwapChain, mRTVDescriptorHeap);

		// Recreate the ray tracing output buffer since when we copy the contento onto the backbuffer for on-screen visualization, sizes must match!
		RecreateRaytracingOutputResource(Width, Height);
	}
}



ComPtr<IDXGIAdapter4> Ray_DX12HardwareRenderer::GetAdapter(bool InUseWarp)
{
	ComPtr<IDXGIFactory4> DXGIFactory;

	u32 CreateFactoryFlags = 0;

#if defined(_DEBUG)
	CreateFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

	ThrowIfFailed(CreateDXGIFactory2(CreateFactoryFlags, IID_PPV_ARGS(&DXGIFactory)));

	ComPtr<IDXGIAdapter1> DXGIAdapter1;
	ComPtr<IDXGIAdapter4> DXGIAdapter4;

	if (InUseWarp)
	{
		ThrowIfFailed(DXGIFactory->EnumWarpAdapter(IID_PPV_ARGS(&DXGIAdapter1)));
		ThrowIfFailed(DXGIAdapter1.As(&DXGIAdapter4));
	}
	else
	{
		SIZE_T MaxDedicatedVideoMemory = 0;
		for (UINT i = 0; DXGIFactory->EnumAdapters1(i, &DXGIAdapter1) != DXGI_ERROR_NOT_FOUND; ++i)
		{
			DXGI_ADAPTER_DESC1 dxgiAdapterDesc1;
			DXGIAdapter1->GetDesc1(&dxgiAdapterDesc1);

			// Check to see if the adapter can create a D3D12 device without actually 
			// creating it. The adapter with the largest dedicated video memory
			// is favored.
			if ((dxgiAdapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
				SUCCEEDED(D3D12CreateDevice(DXGIAdapter1.Get(),
					D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)) &&
				dxgiAdapterDesc1.DedicatedVideoMemory > MaxDedicatedVideoMemory)
			{
				MaxDedicatedVideoMemory = dxgiAdapterDesc1.DedicatedVideoMemory;
				ThrowIfFailed(DXGIAdapter1.As(&DXGIAdapter4));
			}
		}
	}

	return DXGIAdapter4;
}


ComPtr<ID3D12Device5> Ray_DX12HardwareRenderer::CreateDevice(ComPtr<IDXGIAdapter4> InAdapter)
{
	//Ray tracing capable device creation
	ComPtr<ID3D12Device5> d3d12Device5;

	ThrowIfFailed(D3D12CreateDevice(InAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&d3d12Device5)));

	//Here we check if ray tracing is supported by the device
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 d3d12Caps = { };

	//Query the device for the supported set of feature that includes OPTIONS5 and check HRESULT
	ThrowIfFailed(d3d12Device5->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,&d3d12Caps,sizeof(d3d12Caps)));

	//Check if we support ray tracing
	if (d3d12Caps.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
	{
		//Add proper log system here
		//NOTE: this will eventually be replaced by our logging system
		OutputDebugString("Device or Driver does not support ray tracing!");		
		throw std::exception();
	}


	// Enable debug messages in debug mode.
#if defined(_DEBUG)
	ComPtr<ID3D12InfoQueue> pInfoQueue;
	if (SUCCEEDED(d3d12Device5.As(&pInfoQueue)))
	{
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);


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
		NewFilter.DenyList.NumSeverities = _countof(Severities);
		NewFilter.DenyList.pSeverityList = Severities;
		NewFilter.DenyList.NumIDs = _countof(DenyIds);
		NewFilter.DenyList.pIDList = DenyIds;

		ThrowIfFailed(pInfoQueue->PushStorageFilter(&NewFilter));
	}
#endif

	return d3d12Device5;
}


ComPtr<ID3D12CommandQueue> Ray_DX12HardwareRenderer::CreateCommandQueue(ComPtr<ID3D12Device2> InDevice, D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12CommandQueue> D3D12CommandQueue;

	D3D12_COMMAND_QUEUE_DESC desc = {}; 
	desc.Type = InType;
	desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
#if defined(_DEBUG)
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
#else
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
#endif
	desc.NodeMask = 0;

	ThrowIfFailed(InDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&D3D12CommandQueue)));
	

	return D3D12CommandQueue;
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

	// We cache the backbuffer format
	mBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

	DXGI_SWAP_CHAIN_DESC1 SwapChainDesc = {};
	SwapChainDesc.Width = InWidth;
	SwapChainDesc.Height = InHeight;
	SwapChainDesc.Format = mBackBufferFormat; //If we want to let the hw perform gamma correction for us we should use _SRGB format instead
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

	//Get the index of the current swapchain backmbuffer
	mBackBufferIndex = dxgiSwapChain4->GetCurrentBackBufferIndex();

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


ComPtr<ID3D12CommandAllocator> Ray_DX12HardwareRenderer::CreateCommandAllocator( ComPtr<ID3D12Device2> InDevice
																			   , D3D12_COMMAND_LIST_TYPE InType )
{
	ComPtr<ID3D12CommandAllocator> CommandAllocator;
	ThrowIfFailed( InDevice->CreateCommandAllocator(InType, IID_PPV_ARGS(&CommandAllocator)) );
	return CommandAllocator;
}



ComPtr<ID3D12GraphicsCommandList4> Ray_DX12HardwareRenderer::CreateCommandList( ComPtr<ID3D12Device2> InDevice
		                                                                     ,  ComPtr<ID3D12CommandAllocator> InCommandAllocator
		                                                                      , D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12GraphicsCommandList4> CommandList;

	//Create a command list that supports ray tracing
	ThrowIfFailed(InDevice->CreateCommandList(0, InType, InCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&CommandList)));

	//Before to reset a command list, we must close it because it is created in recording state.
	ThrowIfFailed(CommandList->Close());

	return CommandList;
}



void Ray_DX12HardwareRenderer::BeginFrame(float* InClearColor)
{

	// TODO: remove cbuffer update from here (must go in update section of the sample at most) ///
	SceneConstants SceneCB = { };

	SceneCB.CameraPosition = { -150.0f,150.0f,150.0f, 0.0f };
	//SceneCB.CameraPosition = { 0.0f,0.0f,2.0f, 0.0f };

	memcpy(mSceneConstantsCB_DataPtr, &SceneCB, sizeof(SceneConstants));
	//////////////////////////////////////////////////////////////////////////////////////////////


	//Beginning of the frame
	float DefaultClearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
	auto commandAllocator = mD3DCommandAllocators[mBackBufferIndex];
	auto backBuffer = mRenderTargets[mBackBufferIndex];

	//Before any commands can be recorded into the command list, the command allocator and command list needs to be reset to their initial state.
	ThrowIfFailed(commandAllocator->Reset());
	ThrowIfFailed(mD3DCommandList->Reset(commandAllocator.Get(), nullptr));

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


void Ray_DX12HardwareRenderer::Render()
{
	//TODO: We might want to refactor this one as well

	// Get the d3d command list
	auto CommandList = mD3DCommandList.Get();

	// Here we prepare a lamda function that performs a call to DispatchRays
	// Which basically will create a Grid in which each element is a ray 
	auto DispatchRays = [&](auto* commandList, auto* dispatchDesc,auto* stateObject)
	{
		// Since each shader table has only one shader record, the stride is same as the size.
		dispatchDesc->HitGroupTable.StartAddress = mHitGroupShaderTable->GetGPUVirtualAddress();
		dispatchDesc->HitGroupTable.SizeInBytes = mHitGroupShaderTable->GetDesc().Width;
		dispatchDesc->HitGroupTable.StrideInBytes = dispatchDesc->HitGroupTable.SizeInBytes;
		dispatchDesc->MissShaderTable.StartAddress = mMissShaderTable->GetGPUVirtualAddress();
		dispatchDesc->MissShaderTable.SizeInBytes = mMissShaderTable->GetDesc().Width;
		dispatchDesc->MissShaderTable.StrideInBytes = dispatchDesc->MissShaderTable.SizeInBytes;
		dispatchDesc->RayGenerationShaderRecord.StartAddress = mRayGenShaderTable->GetGPUVirtualAddress();
		dispatchDesc->RayGenerationShaderRecord.SizeInBytes = mRayGenShaderTable->GetDesc().Width;
		dispatchDesc->Width = mWidth;
		dispatchDesc->Height = mHeight;
		dispatchDesc->Depth = 1;	
		commandList->SetPipelineState1(stateObject);
		commandList->DispatchRays(dispatchDesc);
	};


	CommandList->SetComputeRootSignature(mRaytracingGlobalRootSignature.Get());

	// Set the descriptor heap(s)
	CommandList->SetDescriptorHeaps(1, mCBVSRVDescriptorHeap.GetAddressOf());
	
	// UAV output buffer is set as a descriptor table
	CommandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, mRaytracingOutputResourceUAVGpuDescriptor);	
	
	// Scene constant buffer
	CommandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::SceneConstantBuffer, mSceneConstantsHandle);
	
	//CommandList->SetComputeRootConstantBufferView(GlobalRootSignatureParams::SceneConstantBuffer, mSceneConstantsCB->GetGPUVirtualAddress());
	
	D3D12_DISPATCH_RAYS_DESC DispatchDesc = {};

	// For each object TLAS trace rays (eventtually move this portion of the code outside)
	// Top level acceleration structure is set as a Shader Resource View
	CommandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::AccelerationStructureSlot, mRayTracingRenderPacket.mTLAS->GetGPUVirtualAddress());

	// Then we are ready to dispatch rays
	DispatchRays(CommandList, &DispatchDesc,mDXRStateObject.Get());
			

	// Done! Copy the result on backbuffer ready to be displayed
	CopyRayTracingOutputToBackBuffer();
}


void Ray_DX12HardwareRenderer::CopyRayTracingOutputToBackBuffer()
{
	auto CommandList = mD3DCommandList.Get();
	auto BackBuffer = mRenderTargets[mBackBufferIndex].Get();

	D3D12_RESOURCE_BARRIER preCopyBarriers[2];
	preCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
	preCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(mRayTracingOutputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	
	CommandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

	CommandList->CopyResource(BackBuffer, mRayTracingOutputBuffer.Get());

	D3D12_RESOURCE_BARRIER postCopyBarriers[2];
	postCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
	postCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(mRayTracingOutputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	CommandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
}


void Ray_DX12HardwareRenderer::EndFrame()
{
	//After transitioning to the correct state, the command list that contains the resource transition barrier must be executed on the command queue.
	ThrowIfFailed(mD3DCommandList->Close());

	ID3D12CommandList* const commandLists[] = { mD3DCommandList.Get() };

	mD3DCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

	u32 SyncInterval = mVSync ? 1 : 0;	
	u32 PresentFlags = IsTearingSupported && !mVSync ? DXGI_PRESENT_ALLOW_TEARING : 0;

	ThrowIfFailed(mSwapChain->Present(SyncInterval, PresentFlags));

	// Wait for the  previous frame to finish
	//WaitForPreviousFrame();

	MoveToNextFrame();
}






