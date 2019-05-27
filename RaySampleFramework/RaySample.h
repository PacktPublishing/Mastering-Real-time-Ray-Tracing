#pragma once

#include "RayPCH.h"
#include "RayWin32Application.h"

class Ray_Sample
{
public:

	Ray_Sample(u32 Width, u32 Height, std::wstring&& SampleName);

	virtual ~Ray_Sample();

	virtual void OnInit() = 0;

	virtual void OnUpdate(float DeltaFrame) = 0;

	virtual void OnRender() = 0;

	virtual void OnDestroy() = 0;

	u32 GetWidth() const { return mWidth; }
 
	u32 GetHeight() const { return mHeight;  }

	const wchar_t* GetSampleName() const { return mSampleName.c_str(); }

	void ResizeWindow(u32 ClientWidth, u32 ClientHeight);
		 
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
