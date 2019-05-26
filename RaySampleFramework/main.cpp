
//Ray Sample Framework Hello World test 
#include "RayPCH.h"
#include "RaySample.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{

	//Create and place the Ray Sample here


	std::wstring SampleName(L"Hello World! Ray Sample");

	Ray_Sample Sample(1280,720,std::move(SampleName));

	return Ray_Win32Application::Run(&Sample, hInstance, nCmdShow);
}