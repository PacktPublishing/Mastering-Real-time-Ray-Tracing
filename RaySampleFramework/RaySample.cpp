
#include "RayPCH.h"
#include "RaySample.h"


Ray_Sample::Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName) : mWidth(Width)
                                                                         , mHeight(Height)
	                                                                     , mSampleName(SampleName)
{



	ResizeWindow(mWidth, mHeight);
}

Ray_Sample::~Ray_Sample()
{

}


void Ray_Sample::ResizeWindow(u32 ClientWidth, u32 ClientHeight)
{
	mWidth = ClientWidth;
	mHeight = ClientHeight;
	mAspectRatio = static_cast<float>(ClientWidth) / static_cast<float>(ClientHeight);
	if (mHardwareRenderer)
	{
		mHardwareRenderer->Resize(mWidth, mHeight);
	}
}

