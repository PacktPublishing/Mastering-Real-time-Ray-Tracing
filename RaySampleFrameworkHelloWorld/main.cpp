
//Ray Sample Framework Hello World test 
#include "../RaySampleFramework/RayPCH.h"
#include "../RaySampleFramework/RayDX12HelloWorldSample.h"


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	
	//Create and place the Ray Sample here 
	std::wstring SampleName(L"RaySampleFramework Hello World! - DX12");

	Ray_DX12HelloWorldSample Sample(1280,720,std::move(SampleName));

	return Ray_Win32Application::Run(&Sample, hInstance, nCmdShow);
}

