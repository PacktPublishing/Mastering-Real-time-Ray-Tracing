#pragma once


#include <fbxsdk.h>
#include <vector>
#include <assert.h>


struct MyVertex
{
	float pos[3];
};


class FBXModelLoader
{

public:

	FBXModelLoader(const FBXModelLoader& rhs) = delete;

	FBXModelLoader& operator=(const FBXModelLoader& rhs) = delete;

	FBXModelLoader& operator=(FBXModelLoader&& rhs) = delete;

	//Singleton
	//In C++11, the following is guaranteed to perform thread-safe initialisation
	static FBXModelLoader& Get()
	{
		static FBXModelLoader Instance;
		return Instance;
	}

	bool Load(const char* PathToFile,std::vector<MyVertex>& OutVertexVector,std::vector<u16>& OutIndices)
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


		FbxNode* FbxRootNode = pFbxScene->GetRootNode();

		// Let's traverse the scene and load our model
		TraverseScene(FbxRootNode,OutVertexVector,OutIndices,ExtractMeshData);
	
	
		return true;
	}

private:

	// We extract just vertices 
	static void ExtractMeshData(FbxMesh* Mesh, std::vector<MyVertex>& OutVertexVector, std::vector<u16>& OutIndices)
	{
		i32 NumVertices = Mesh->GetControlPointsCount();
		OutVertexVector.reserve(NumVertices);

		// Read Vertices
		for (i32 j = 0; j < NumVertices; j++)
		{

			FbxVector4 Coord = Mesh->GetControlPointAt(j);

			MyVertex vertex;

			//We flip the z-axis with the y because our default up axis is y!
			vertex.pos[0] = (float)Coord.mData[0];
			vertex.pos[2] = (float)Coord.mData[1];
			vertex.pos[1] = (float)Coord.mData[2];
			OutVertexVector.push_back(vertex);
		}


		//Read Indices
		i32 TriangleCount = Mesh->GetPolygonCount();
		u32 IndexCount    = TriangleCount*3;
		OutIndices.reserve(IndexCount);
		for (i32 i = 0; i < TriangleCount; ++i)
		{
			for (i32 j = 2; j >= 0; --j)
			{
				i32 ctrlPointIndex = Mesh->GetPolygonVertex(i, j);
				OutIndices.push_back(ctrlPointIndex);
			}
		}

	}

	void TraverseScene(FbxNode* Node,std::vector<MyVertex>& OutVertexVector,std::vector<u16>& OutIndices,void (*ExtractMeshDataCallback)(FbxMesh*,std::vector<MyVertex>&,std::vector<u16>&))
	{

		if (Node == nullptr)
		{
			return;
		}

		FbxMesh* Mesh = Node->GetMesh();
		if (Mesh != nullptr)
		{
			ExtractMeshDataCallback(Mesh,OutVertexVector, OutIndices);
		}

		for (i32 i = 0; i < Node->GetChildCount(); i++)
		{
			TraverseScene(Node->GetChild(i),OutVertexVector,OutIndices,ExtractMeshDataCallback);
		}

	}

	FbxManager* mFbxSdkManager = nullptr;

	FBXModelLoader() = default;


};








