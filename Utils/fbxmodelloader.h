#pragma once

#include <fbxsdk.h>
#include <vector>
#include <assert.h>


struct MyVertex
{
	float pos[3];
	float n[3];
	float uv[2];
};


struct SMeshSection
{

	SMeshSection() = default;

	~SMeshSection()
	{

		if (mIndexBuffer)
		{
			// TODO: Handle index buffer deallocation (I was getting a strange crash on the pointer. Looking for a better way to handle this)
			//delete [] ((u16*)mIndexBuffer);
			mIndexBuffer = nullptr;
			mIndexCount = 0;
		}
	}


	void* mIndexBuffer = nullptr;
	int   mIndexCount = 0;
	int   mMaterialID = -1;
};

struct SMesh
{
	SMesh() = default;

	std::vector<SMeshSection> mMeshSections;
	std::vector<MyVertex> mVertexBuffer;
	bool  mUse32BitIndices = false;
};


struct SModel
{
	SModel() = default;

	std::vector<SMesh> mMeshes;
};


class FBXModelLoader
{

public:

	FBXModelLoader(const FBXModelLoader& rhs) = delete;

	FBXModelLoader& operator=(const FBXModelLoader& rhs) = delete;

	FBXModelLoader& operator=(FBXModelLoader&& rhs) = delete;

	// Singleton
	// In C++11, the following is guaranteed to perform thread-safe initialisation
	static FBXModelLoader& Get()
	{
		static FBXModelLoader Instance;
		return Instance;
	}

	bool Load(const char* PathToFile, SModel& OutModel,bool bNoMeshSections = false)
	{
		if (mFbxSdkManager == nullptr)
		{
			mFbxSdkManager = FbxManager::Create();

			FbxIOSettings* pIOsettings = FbxIOSettings::Create(mFbxSdkManager, IOSROOT);
			mFbxSdkManager->SetIOSettings(pIOsettings);
		}

		FbxImporter* pImporter = FbxImporter::Create(mFbxSdkManager, "");
		FbxScene* pFbxScene = FbxScene::Create(mFbxSdkManager, "");

		bool bSuccess = pImporter->Initialize(PathToFile, -1, mFbxSdkManager->GetIOSettings());
		if (!bSuccess)
		{
			return false;
		}

		bSuccess = pImporter->Import(pFbxScene);
		if (!bSuccess)
		{
			return false;
		}

		pImporter->Destroy();


		FbxNode* Node = pFbxScene->GetRootNode();

		// Let's traverse the scene and load our model
		const int ChildCount = Node->GetChildCount();
		for (int c = 0; c < ChildCount; ++c)
		{
			TraverseScene(Node->GetChild(c), OutModel, bNoMeshSections, ExtractMeshData);
		}


		return true;
	}

private:


	static FbxVector4 MultT(FbxNode* Node, FbxVector4 vector) 
	{
		FbxAMatrix matrixGeo;
		matrixGeo.SetIdentity();
		if (Node->GetNodeAttribute())
		{
			const FbxVector4 lT = Node->GetGeometricTranslation(FbxNode::eSourcePivot);
			const FbxVector4 lR = Node->GetGeometricRotation(FbxNode::eSourcePivot);
			const FbxVector4 lS = Node->GetGeometricScaling(FbxNode::eSourcePivot);
			matrixGeo.SetT(lT);
			matrixGeo.SetR(lR);
			matrixGeo.SetS(lS);
		}

		FbxAMatrix globalMatrix = Node->EvaluateLocalTransform();

		FbxAMatrix matrix = globalMatrix * matrixGeo;
		FbxVector4 result = matrix.MultT(vector);
		
		return result;
	}


	static FbxLayerElementArrayTemplate<int>* GetMaterialIndices(FbxMesh* pMesh)
	{
		//SmartPrintf("LayerCount: %d\n", pMesh->GetLayerCount());
		FbxLayerElementArrayTemplate<int>* TmpArray = nullptr;
		if (pMesh->GetLayerCount() >= 1)
		{
			FbxLayerElementMaterial* LayerMaterial = pMesh->GetLayer(0)->GetMaterials();
			if (LayerMaterial)
			{
				FbxLayerElementArrayTemplate<int>* TmpArray = &LayerMaterial->GetIndexArray();
				//SmartPrintf("ArrayCount: %d\n", TmpArray->GetCount());
				//for (int i = 0; i < TmpArray->GetCount(); i++)
				//{
				//	SmartPrintf("%d ", TmpArray->GetAt(i));
				//}
				//SmartPrintf("\n");
				return TmpArray;
			}
			return nullptr;
		}
		return nullptr;
	}


	// We extract just vertices 
	static void ExtractMeshData(FbxMesh* Mesh,FbxNode* Node, SMesh& OutMesh, bool bNoMeshSections)
	{
		const int UVCount = Mesh->GetElementUVCount();
		const int NormalCount = Mesh->GetElementNormalCount();
		const int TangentCount = Mesh->GetElementTangentCount();
		const int NumVertices = Mesh->GetControlPointsCount();
		const int TriangleCount = Mesh->GetPolygonCount();

		const bool HasNormal = NormalCount > 0;
		const bool HasUV = UVCount > 0;
		const bool HasTangent = TangentCount > 0;

		// Reserve memory for vertices 
		OutMesh.mVertexBuffer.reserve(NumVertices);

		// Get material indices/count
		auto MaterialIndices = GetMaterialIndices(Mesh);

		// Read Vertices
		for (int j = 0; j < NumVertices; j++)
		{

			FbxVector4 Coord = Mesh->GetControlPointAt(j);

			Coord = MultT(Node,Coord);

			MyVertex vertex;

			// We flip the z-axis with the y because our default up axis is y!
			vertex.pos[0] = (float)Coord[0];
			vertex.pos[1] = (float)Coord[1];
			vertex.pos[2] = (float)Coord[2];
			OutMesh.mVertexBuffer.push_back(vertex);
		}

		// Extract this mesh elements		
		bool Use32BitIndices = (NumVertices > 65536);

		OutMesh.mUse32BitIndices = Use32BitIndices;

		const auto MeshSectionCount = OutMesh.mMeshSections.size();

		std::vector< std::vector<unsigned int> > TempIndexArray(MeshSectionCount);

		for (int PolyIndex = 0; PolyIndex < TriangleCount; ++PolyIndex)
		{
			// Index of a given mesh section with a given material			
			const int MeshSectionIndex = MaterialIndices ? MaterialIndices->GetAt(PolyIndex) : 0; 	
			assert(MeshSectionIndex < OutMesh.mMeshSections.size() || bNoMeshSections);

			for (int VtxIdx = 0; VtxIdx < 3; ++VtxIdx)
			{
				// Read Indices	
				const int Index = Mesh->GetPolygonVertex(PolyIndex, VtxIdx);
				TempIndexArray[bNoMeshSections ? 0 : MeshSectionIndex].push_back(Index);

				// Read UVs
				FbxVector2 OutUV;
				bool bUnmapped;
				for (int u = 0; u < UVCount; ++u)
				{
					auto ElementUV = Mesh->GetElementUV(u);
					if (ElementUV && Mesh->GetPolygonVertexUV(PolyIndex, VtxIdx, ElementUV->GetName(), OutUV, bUnmapped))
					{
						OutMesh.mVertexBuffer[Index].uv[0] = bUnmapped ? 0.f : (float)OutUV[0];
						OutMesh.mVertexBuffer[Index].uv[1] = bUnmapped ? 0.f : (float)OutUV[1];
					}
				}

				// Read Normals
				FbxVector4 normal;
				if (Mesh->GetPolygonVertexNormal(PolyIndex, VtxIdx, normal))
				{
					OutMesh.mVertexBuffer[Index].n[0] = (float)normal[0];
					OutMesh.mVertexBuffer[Index].n[1] = (float)normal[1];
					OutMesh.mVertexBuffer[Index].n[2] = (float)normal[2];
				}
			}
		}


		// Copy indices back to each mesh section
		unsigned int MeshSectionIndex = 0;
		for (auto& MeshSection : OutMesh.mMeshSections)
		{
			MeshSection.mMaterialID = MeshSectionIndex;
			size_t IndexCount = TempIndexArray[bNoMeshSections ? 0 : MeshSectionIndex].size();

			//assert((IndexCount % 3) == 0);
			//u32 lol = 1;
			//for (u32 idx=0;idx<IndexCount;idx+=3)
			//{
			//	if (lol % 2)
			//	{
			//		std::swap(TempIndexArray[MeshSectionIndex][idx], TempIndexArray[MeshSectionIndex][idx + 2]);
			//	}
			//	++lol;
			//}


			MeshSection.mIndexCount = (int)IndexCount;
			if (Use32BitIndices)
			{
				MeshSection.mIndexBuffer = new unsigned int[IndexCount];
			}
			else
			{
				MeshSection.mIndexBuffer = new unsigned short[IndexCount];
			}

			for (size_t i = 0; i < IndexCount; ++i)
			{
				if (Use32BitIndices)
				{
					((unsigned int*)MeshSection.mIndexBuffer)[i] = TempIndexArray[MeshSectionIndex][i];
				}
				else
				{
					((unsigned short*)MeshSection.mIndexBuffer)[i] = TempIndexArray[MeshSectionIndex][i];
				}

			}
			++MeshSectionIndex;
		}


	}

	void TraverseScene(FbxNode* Node, SModel& OutModel,bool bNoMeshSections,void(*ExtractMeshDataCallback)(FbxMesh*, FbxNode*, SMesh&,bool))
	{

		if (Node == nullptr)
		{
			return;
		}

		auto Attr = Node->GetNodeAttribute();
		if (!Attr)
		{
			return;
		}

		switch (Attr->GetAttributeType()) {
		case FbxNodeAttribute::eMesh:
		{
			FbxMesh* Mesh = Node->GetMesh();
			if (Mesh != nullptr)
			{
				// Get the total number of materials
				const size_t MaterialCount = Node->GetMaterialCount();				

				// Create our internal mesh
				SMesh mesh;

				// We have as many mesh sections as are the total number of materials. If we don't have materials, let's create at list one mesh section.
				mesh.mMeshSections.resize((MaterialCount == 0 || bNoMeshSections) ? 1 : MaterialCount);

				// We extract the data from the FBX
				ExtractMeshDataCallback(Mesh,Node,mesh,bNoMeshSections);

				// After we've created our internal mesh, we store it into our model struct 
				OutModel.mMeshes.push_back(mesh);
			}
		}
		break;
		default:
			break;
		}

		const int ChildCount = Node->GetChildCount();
		for (int i = 0; i < ChildCount; i++)
		{
			TraverseScene(Node->GetChild(i), OutModel, bNoMeshSections, ExtractMeshDataCallback);
		}

	}

	FbxManager* mFbxSdkManager = nullptr;

	FBXModelLoader() = default;


};

