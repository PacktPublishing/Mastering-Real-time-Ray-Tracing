#pragma once

#include "RayPCH.h"
#include "RayWin32Application.h"

class Ray_Sample
{
public:

	Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName);

	virtual ~Ray_Sample();

protected:

	/** Viewport relevant data members */
	u32 mWidth;
	
	u32 mHeight;

	float mAspectRatio = 0.0f;


	/** Add device related pointer to implementation */

private:




	/** The name of the sample that is shown on the window title */
	std::wstring mSampleName;
};
