#pragma once


#include "RayPCH.h"

//Forward declaration
class Ray_Sample;

class Ray_Win32Application
{

public:

	/** Call this method to start your Ray sample */
	static i32 Run(Ray_Sample* pSample, HINSTANCE hInstance, int nCmdShow);

	/** Window handle getter */
	static HWND GetWindowHandle() { return mWindowHandle; }

protected:

	/** The application window procedure */
	static LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	/** Toggle fullscreen */
	static void ToggleFullscreen();

private:

	//TODO: We might need to implement a delegate solution to avoid having to point to Ray_Sample
	static Ray_Sample* mCachedSamplePtr;

	/** The handle to the window to which we will render to */
	static HWND mWindowHandle;

	/** Hold whether we are in fullscreen mode or not */
	static bool mFullscreen;

	/** Is this app running */
	static bool mAppIsRunning;

	/** Do we want for vsync to be enabled ? */
	static bool mVSync;

	/** Current Window Rect */
	static RECT mWindowRect;

};
