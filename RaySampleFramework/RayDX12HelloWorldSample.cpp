
#include "RayPCH.h"
#include "RayDX12HelloWorldSample.h"
#include "RayDX12HardwareRenderer.h"

Ray_DX12HelloWorldSample::Ray_DX12HelloWorldSample(u32 Width, u32 Height, std::wstring&& SampleName) : Ray_Sample(Width,Height,std::move(SampleName))
{
	//init the DX12 hardware renderer
	mHardwareRenderer = std::make_unique<Ray_DX12HardwareRenderer>();
}

Ray_DX12HelloWorldSample::~Ray_DX12HelloWorldSample()
{


}

void Ray_DX12HelloWorldSample::OnInit()
{
	if (mHardwareRenderer)
	{

		mHardwareRenderer->Init(mWidth, mHeight, Ray_Win32Application::GetWindowHandle() , mUseWarp);

	}
}

void Ray_DX12HelloWorldSample::OnUpdate(float DeltaFrame)
{


}

void Ray_DX12HelloWorldSample::OnRender()
{

	if (mHardwareRenderer)
	{

		float ClearColor[] = {1.0f,0.0f,0.0f,1.0f};

		//Beginning of the frame
		mHardwareRenderer->BeginFrame();



		//Perform draws/dispatch rays/dispatch here
		auto DX12HRenderer = static_cast<Ray_DX12HardwareRenderer*>(mHardwareRenderer.get());
		auto CmdList = DX12HRenderer->GetCommandList();
		

		//Render the actual frame 
		mHardwareRenderer->Render();
		


		// Present
		mHardwareRenderer->EndFrame();

	}
	else
	{
		//TODO: place Log here
	}
}

void Ray_DX12HelloWorldSample::OnDestroy()
{

	if (mHardwareRenderer)
	{
		mHardwareRenderer->Destroy();
	}
	else
	{
		//TODO: place Log here
	}

}
