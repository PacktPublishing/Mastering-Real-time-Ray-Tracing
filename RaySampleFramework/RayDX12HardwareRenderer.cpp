
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




void Ray_DX12HardwareRenderer::FlushGPU()
{
	for (size_t i = 0; i < kMAX_BACK_BUFFER_COUNT; i++)
	{
		u64 FenceValueForSignal = ++mFrameFenceValues[i];
		mD3DCommandQueue->Signal(mFences[i].Get(), FenceValueForSignal);
		if (mFences[i]->GetCompletedValue() < mFrameFenceValues[i])
		{
			mFences[i]->SetEventOnCompletion(FenceValueForSignal, mFenceEvent);
			WaitForSingleObject(mFenceEvent, INFINITE);
		}
	}
	mBackBufferIndex = 0;
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
void Ray_DX12HardwareRenderer::SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& Desc, ComPtr<ID3D12RootSignature>* RootSig)
{
	ComPtr<ID3DBlob> Blob;
	ComPtr<ID3DBlob> Error;
	ThrowIfFailed(D3D12SerializeRootSignature(&Desc, D3D_ROOT_SIGNATURE_VERSION_1, &Blob, &Error) , Error ? static_cast<wchar_t*>(Error->GetBufferPointer()) : nullptr);
	ThrowIfFailed(mD3DDevice->CreateRootSignature(1, Blob->GetBufferPointer(), Blob->GetBufferSize(), IID_PPV_ARGS(&(*RootSig))));
}

// Root signatures represent parameters passed to shaders 
void Ray_DX12HardwareRenderer::CreateRootSignatures()
{

	// Global Root Signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	{
		CD3DX12_DESCRIPTOR_RANGE UAVDescriptor;
		UAVDescriptor.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
		CD3DX12_ROOT_PARAMETER RootParameters[GlobalRootSignatureParams::Count];
		RootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &UAVDescriptor);
		RootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
		CD3DX12_ROOT_SIGNATURE_DESC GlobalRootSignatureDesc(ARRAYSIZE(RootParameters), RootParameters);
		SerializeAndCreateRaytracingRootSignature(GlobalRootSignatureDesc, &mRaytracingGlobalRootSignature);
	}

	// Local Root Signature
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	{
		CD3DX12_ROOT_PARAMETER RootParameters[LocalRootSignatureParams::Count];
		RootParameters[LocalRootSignatureParams::ViewportConstantSlot].InitAsConstants(SizeOfInUint32(mRayGenCB), 0, 0);
		CD3DX12_ROOT_SIGNATURE_DESC LocalRootSignatureDesc(ARRAYSIZE(RootParameters), RootParameters);
		LocalRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		SerializeAndCreateRaytracingRootSignature(LocalRootSignatureDesc, &mRaytracingLocalRootSignature);
	}

}



// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void Ray_DX12HardwareRenderer::CreateLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* RaytracingPipeline)
{
	// Hit group and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

	// Local root signature to be used in a ray gen shader.
	{
		auto LocalRootSignature = RaytracingPipeline->CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
		LocalRootSignature->SetRootSignature(mRaytracingLocalRootSignature.Get());
		
		// Shader association
		auto RootSignatureAssociation = RaytracingPipeline->CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		RootSignatureAssociation->SetSubobjectToAssociate(*LocalRootSignature);
		RootSignatureAssociation->AddExport(gRaygenShaderName);
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
	UINT PayloadSize = 4 * sizeof(float);   // float4 color
	UINT AttributeSize = 2 * sizeof(float); // float2 barycentrics
	ShaderConfig->Config(PayloadSize, AttributeSize);


	// Local root signature and shader association
	CreateLocalRootSignatureSubobjects(&RaytracingPipeline);
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.

	// Global root signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	auto GlobalRootSignature = RaytracingPipeline.CreateSubobject<CD3D12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	GlobalRootSignature->SetRootSignature(mRaytracingGlobalRootSignature.Get());

	// Pipeline config
	// Defines the maximum TraceRay() recursion depth.
	auto PipelineConfig = RaytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	
	// PERFOMANCE TIP: Set max recursion depth as low as needed 
	// as drivers may apply optimization strategies for low recursion depths. 
	u32 MaxRecursionDepth = 1; // ~ primary rays only. 
	PipelineConfig->Config(MaxRecursionDepth);


	// TODO: mDXRStateObject is of type ID3D12StateObjectPrototype. Please verify that this object is not temporary in relation to the DXR API definition
	ThrowIfFailed(mD3DDevice->CreateStateObject(RaytracingPipeline, IID_PPV_ARGS(&mDXRStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

// Create a heap for descriptors for ray tracing resource 
void Ray_DX12HardwareRenderer::CreateDescriptorHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC DescriptorHeapDesc = {};
	
	// Allocate a heap for 3 descriptors:
	// 2 - bottom and top level acceleration structure fallback wrapped pointers
	// 1 - raytracing output texture SRV
	DescriptorHeapDesc.NumDescriptors = 3;
	DescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	DescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	DescriptorHeapDesc.NodeMask = 0;
	mD3DDevice->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(&mDescriptorHeap));
	NAME_D3D12_OBJECT(mDescriptorHeap);

	mDescriptorSize = mD3DDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

}

// TODO: we build geometry in the hardware renderer for now, but we might move this elsewhere due to a refactor
// Build geometry 
void Ray_DX12HardwareRenderer::BuildGeometry()
{
	// Get the currnt d3d device
	auto Device = mD3DDevice.Get();

	// Loads vertices and indices from FBX file
	std::vector<MyVertex> VBuffer;
	std::vector<u16> IBuffer;
	FBXModelLoader::Get().Load((gAssetRootDir + "Sphere.fbx").c_str(), VBuffer, IBuffer);

	// This helper functions creates a buffer resource of type committed and upload vertex and index data in each one of them
	AllocateUploadBuffer(Device, &VBuffer[0], sizeof(MyVertex)*VBuffer.size(), &mVB);

	AllocateUploadBuffer(Device, &IBuffer[0], IBuffer.size()*sizeof(u16), &mIB);
}

// Build acceleration structures 
void Ray_DX12HardwareRenderer::BuildAccelerationStructures()
{
	// Get current command allocator given the index of the current frame backbuffer
	auto CommnadAllocator = mD3DCommandAllocator[mBackBufferIndex].Get();
	auto Device = mD3DDevice.Get();

	// Reset the command list for the acceleration structure construction.
	mD3DCommandList->Reset(CommnadAllocator, nullptr);

	D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
	geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geometryDesc.Triangles.IndexBuffer = mIB->GetGPUVirtualAddress();
	geometryDesc.Triangles.IndexCount = static_cast<UINT>(mIB->GetDesc().Width) / sizeof(u16);
	geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
	geometryDesc.Triangles.Transform3x4 = 0;
	geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
	geometryDesc.Triangles.VertexCount = static_cast<UINT>(mVB->GetDesc().Width) / (sizeof(float)*3);
	geometryDesc.Triangles.VertexBuffer.StartAddress = mVB->GetGPUVirtualAddress();
	geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(MyVertex);

	// Mark the geometry as opaque. 
	// PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
	// Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
	geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	// Get required sizes for an acceleration structure.
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs = {};
	topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	topLevelInputs.Flags = buildFlags;
	topLevelInputs.NumDescs = 1;
	topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
	mD3DDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
	
	ThrowIfFalse(topLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS BottomLevelInputs = topLevelInputs;
	BottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	BottomLevelInputs.pGeometryDescs = &geometryDesc;
	mD3DDevice->GetRaytracingAccelerationStructurePrebuildInfo(&BottomLevelInputs, &bottomLevelPrebuildInfo);
	ThrowIfFalse(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

	ComPtr<ID3D12Resource> scratchResource;
	AllocateUAVBuffer(Device, std::max(topLevelPrebuildInfo.ScratchDataSizeInBytes, bottomLevelPrebuildInfo.ScratchDataSizeInBytes), &scratchResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");

	// Allocate resources for acceleration structures.
	// Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
	// Default heap is OK since the application doesn’t need CPU read/write access to them. 
	// The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
	// and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
	//  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
	//  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
	{
		D3D12_RESOURCE_STATES InitialResourceState;
		InitialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
		
		AllocateUAVBuffer(Device, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, &mBLAS, InitialResourceState, L"BottomLevelAccelerationStructure");
		AllocateUAVBuffer(Device, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &mTLAS, InitialResourceState, L"TopLevelAccelerationStructure");
	}


    // Ray tracing instance desc needs BLAS virtual GPU address
	ComPtr<ID3D12Resource> InstanceDescs;
	{
		D3D12_RAYTRACING_INSTANCE_DESC InstanceDesc = {};
		InstanceDesc.Transform[0][0] = InstanceDesc.Transform[1][1] = InstanceDesc.Transform[2][2] = 1;
		InstanceDesc.InstanceMask = 1;
		InstanceDesc.AccelerationStructure = mBLAS->GetGPUVirtualAddress();
		AllocateUploadBuffer(Device, &InstanceDesc, sizeof(InstanceDesc), &InstanceDescs, L"InstanceDescs");
	}

	// Bottom Level Acceleration Structure desc
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC BottomLevelBuildDesc = {};
	{
		BottomLevelBuildDesc.Inputs = BottomLevelInputs;
		BottomLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
		BottomLevelBuildDesc.DestAccelerationStructureData = mBLAS->GetGPUVirtualAddress();
	}

	// Top Level Acceleration Structure desc
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
	{
		topLevelInputs.InstanceDescs = InstanceDescs->GetGPUVirtualAddress();
		topLevelBuildDesc.Inputs = topLevelInputs;
		topLevelBuildDesc.DestAccelerationStructureData = mTLAS->GetGPUVirtualAddress();
		topLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
	}

	auto BuildAccelerationStructure = [&](auto* RaytracingCommandList)
	{
		RaytracingCommandList->BuildRaytracingAccelerationStructure(&BottomLevelBuildDesc, 0, nullptr);
		mD3DCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(mBLAS.Get()));
		RaytracingCommandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
	};

	// Build acceleration structure.
	BuildAccelerationStructure(mD3DCommandList.Get());
	
	// Kick off acceleration structure construction.
	ThrowIfFailed(mD3DCommandList->Close());
	ID3D12CommandList *commandLists[] = { mD3DCommandList.Get() };
	mD3DCommandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

	// Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
	FlushGPU();

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
	auto DescriptorHeapCpuBase = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
	if (DescriptorIndexToUse >= mDescriptorHeap->GetDesc().NumDescriptors)
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
	
	mRaytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(mDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), mRaytracingOutputResourceUAVDescriptorHeapIndex, mDescriptorSize);
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


	mRaytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(mDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), mRaytracingOutputResourceUAVDescriptorHeapIndex, mDescriptorSize);
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
	//mD3DCommandQueue.Get()->SetName(L"LOL");

	//Swap chain creation
	mSwapChain = CreateSwapChain(InHwnd, mD3DCommandQueue, InWidth, InHeight, kMAX_BACK_BUFFER_COUNT);

	//Get the index of the current swapchain backmbuffer
	mBackBufferIndex = mSwapChain->GetCurrentBackBufferIndex();

	//Create a descriptor heap of type RTV for the swap chain back buffers
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


	// Create fences for GPU flush
	for (u32 i=0;i<kMAX_BACK_BUFFER_COUNT;++i)
	{
		mFences[i] = CreateFence(mD3DDevice);;
	}


	// Create the dx12 fence
	mFence = CreateFence(mD3DDevice);
	
	//Create the CPU event that we'll use to stall the CPU on the fence value (the fence value will get signaled from the GPU as soon as the GPU will reach the fence)	
	mFenceEvent = CreateEventHandle();

	
	// TODO: A heavy refactoring is needed for the ray tracing object creation and management part.
	// In particular we expect to manage heap descriptors, PSO etc. in separated systems/classes. This should promote more modularity and let the code to be more maintainable.

	// Ray Tracing objects init

    // Create root signatures for the shaders.
	CreateRootSignatures();

	// Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
	CreateRaytracingPipelineStateObject();

	// Create a heap for descriptors.
	CreateDescriptorHeap();

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
	//WaitForPreviousFrame();

	//Close the CPU event 
	CloseHandle(mFenceEvent);

	//Reset index and vertex buffer resources
	mIB.Reset();

	mVB.Reset();

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


//ComPtr<ID3D12Device2> Ray_DX12HardwareRenderer::CreateDevice(ComPtr<IDXGIAdapter4> InAdapter)
ComPtr<ID3D12Device5> Ray_DX12HardwareRenderer::CreateDevice(ComPtr<IDXGIAdapter4> InAdapter)
{
	//ID3D12Device2 is not for ray tracing with DXR
	//ComPtr<ID3D12Device2> d3d12Device2;

	//ThrowIfFailed(D3D12CreateDevice(InAdapter.Get(),D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device2)));

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
	//if (SUCCEEDED(d3d12Device2.As(&pInfoQueue)))
	if (SUCCEEDED(d3d12Device5.As(&pInfoQueue)))
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

	//return d3d12Device2;
	return d3d12Device5;
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



ComPtr<ID3D12GraphicsCommandList4> Ray_DX12HardwareRenderer::CreateCommandList(ComPtr<ID3D12Device2> InDevice
		, ComPtr<ID3D12CommandAllocator> InCommandAllocator
		, D3D12_COMMAND_LIST_TYPE InType)
{
	ComPtr<ID3D12GraphicsCommandList4> CommandList;

	//Create a command list that supports ray tracing
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

void Ray_DX12HardwareRenderer::Render()
{
	//TODO: We might want to refactor this one as well

	// Get the d3d command list
	auto CommandList = mD3DCommandList.Get();

	// Here we prepare a lamda function that performs a call to DispatchRays
	// Which basically will create a Grid in which each element is a ray 
	auto DispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc)
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

	D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
	CommandList->SetDescriptorHeaps(1, mDescriptorHeap.GetAddressOf());
	CommandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, mRaytracingOutputResourceUAVGpuDescriptor);
	CommandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::AccelerationStructureSlot, mTLAS->GetGPUVirtualAddress());
	DispatchRays(CommandList, mDXRStateObject.Get(), &dispatchDesc);


	// Done! Copy the result on backbuffer ready to be displayed
	CopyRayTracingOutputToBackBuffer();
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
	WaitForPreviousFrame();
}






