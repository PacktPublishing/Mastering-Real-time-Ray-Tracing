
#include "RayPCH.h"
#include "RaySample.h"


Ray_Sample::Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName) : mWidth(Width)
                                                                         , mHeight(Height)
	                                                                     , mSampleName(SampleName)
{

	
}

Ray_Sample::~Ray_Sample()
{

}

