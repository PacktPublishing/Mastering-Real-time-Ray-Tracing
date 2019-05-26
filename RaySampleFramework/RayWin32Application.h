#pragma once


#include "RayPCH.h"

//Forward declaration
class Ray_Sample;

class Ray_Win32Application
{

public:

	/** Call this method to start your Ray sample */
	static int Run(Ray_Sample* pSample, HINSTANCE hInstance, int nCmdShow);

	/** Window handle getter */
	static HWND GetWindowHandle() { return mWindowHandle; }

private:

	/** The handle to the window to which we will render to */
	static HWND mWindowHandle;

	/** Hold whether we are in fullscreen mode or not */
	static bool mFullscreen;

};
