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

	bool Load(const char* PathToFile,std::vector<MyVertex>* pOutVertexVector)
	{
		if (mFbxSdkManager == nullptr)
		{
			mFbxSdkManager = FbxManager::Create();

			FbxIOSettings* pIOsettings = FbxIOSettings::Create(mFbxSdkManager, IOSROOT);
			mFbxSdkManager->SetIOSettings(pIOsettings);
		}

		FbxImporter* pImporter = FbxImporter::Create(mFbxSdkManager, "");
		FbxScene* pFbxScene = FbxScene::Create(mFbxSdkManager, "");

		//bool bSuccess = pImporter->Initialize("C:\\MyPath\\MyModel.fbx", -1, mFbxSdkManager->GetIOSettings());

		bool bSuccess = pImporter->Initialize(PathToFile, -1, mFbxSdkManager->GetIOSettings());
		if (!bSuccess) return false;

		bSuccess = pImporter->Import(pFbxScene);
		if (!bSuccess) return false;

		pImporter->Destroy();

		FbxNode* pFbxRootNode = pFbxScene->GetRootNode();

		if (pFbxRootNode)
		{
			for (int i = 0; i < pFbxRootNode->GetChildCount(); i++)
			{
				FbxNode* pFbxChildNode = pFbxRootNode->GetChild(i);

				if (pFbxChildNode->GetNodeAttribute() == NULL)
					continue;

				FbxNodeAttribute::EType AttributeType = pFbxChildNode->GetNodeAttribute()->GetAttributeType();

				if (AttributeType != FbxNodeAttribute::eMesh)
					continue;

				FbxMesh* pMesh = (FbxMesh*)pFbxChildNode->GetNodeAttribute();

				FbxVector4* pVertices = pMesh->GetControlPoints();
				

				for (int j = 0; j < pMesh->GetPolygonCount(); j++)
				{
					int iNumVertices = pMesh->GetPolygonSize(j);
					assert(iNumVertices == 3);

					for (int k = 0; k < iNumVertices; k++) {
						int iControlPointIndex = pMesh->GetPolygonVertex(j, k);

						MyVertex vertex;
						vertex.pos[0] = (float)pVertices[iControlPointIndex].mData[0];
						vertex.pos[1] = (float)pVertices[iControlPointIndex].mData[1];
						vertex.pos[2] = (float)pVertices[iControlPointIndex].mData[2];
						pOutVertexVector->push_back(vertex);
					}
				}

			}

		}
		return true;
	}

private:

	FbxManager* mFbxSdkManager = nullptr;

	FBXModelLoader() = default;


};








