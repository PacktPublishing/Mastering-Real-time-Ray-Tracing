
#include "RayPCH.h"
#include "RayWin32Application.h"
#include "RaySample.h"
#include "FrameManager.h"

HWND Ray_Win32Application::mWindowHandle = nullptr;

//TODO: remove sample cached pointer. Favour a delegate implementation instead
Ray_Sample* Ray_Win32Application::mCachedSamplePtr = nullptr;

bool Ray_Win32Application::mAppIsRunning = false;

bool Ray_Win32Application::mFullscreen = false;

RECT Ray_Win32Application::mWindowRect;

//Frame manager is used only in this translation unit
static FrameManager gFrameManager;

static void ParseCommandLineArguments(Ray_Sample* InRaySample)
{
	if (InRaySample)
	{
		int32_t argc;
		wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);

		for (int32_t i = 0; i < argc; ++i)
		{
			if (wcscmp(argv[i], L"-w") == 0 || wcscmp(argv[i], L"--width") == 0)
			{
				auto Width = wcstol(argv[++i], nullptr, 10);
				InRaySample->SetWidth(Width);
			}
			if (wcscmp(argv[i], L"-h") == 0 || wcscmp(argv[i], L"--height") == 0)
			{
				auto Height = wcstol(argv[++i], nullptr, 10);
				InRaySample->SetHeight(Height);
			}
			if (wcscmp(argv[i], L"-warp") == 0 || wcscmp(argv[i], L"--warp") == 0)
			{
				InRaySample->SetUseWarp(true);
			}
		}

		// Free memory allocated by CommandLineToArgvW
		LocalFree(argv);
	}
}


static HWND CreateRenderWindow( const wchar_t* InWindowClassName
	                          , HINSTANCE InhInst
	                          , const wchar_t* InWindowTitle
	                          , u32 InWidth
	                          , u32 InHeight
	                          , WNDPROC InWndProc)
{
	//Register a window class for our render window
	WNDCLASSEXW windowClass = { 0 };

	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = InWndProc;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;
	windowClass.hInstance = InhInst;
	windowClass.hIcon = LoadIcon(InhInst, NULL);
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	windowClass.lpszMenuName = NULL;
	windowClass.lpszClassName = InWindowClassName;
	windowClass.hIconSm = LoadIcon(InhInst, NULL);

	static ATOM atom = RegisterClassExW(&windowClass);
	assert(atom > 0);

	//Create window class
	int screenWidth = GetSystemMetrics(SM_CXSCREEN);
	int screenHeight = GetSystemMetrics(SM_CYSCREEN);

	RECT windowRect = { 0, 0, static_cast<LONG>(InWidth), static_cast<LONG>(InHeight) };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	int windowWidth = windowRect.right - windowRect.left;
	int windowHeight = windowRect.bottom - windowRect.top;

	// Center the window within the screen. Clamp to 0, 0 for the top-left corner.
	int windowX = std::max<int>(0, (screenWidth - windowWidth) / 2);
	int windowY = std::max<int>(0, (screenHeight - windowHeight) / 2);

	//Now create the actual window
	HWND hWnd = CreateWindowExW(0,		
		InWindowClassName,
		InWindowTitle,
		WS_OVERLAPPEDWINDOW,
		windowX,
		windowY,
		windowWidth,
		windowHeight,
		NULL,
		NULL,
		InhInst,
		nullptr
	);

	assert(hWnd && "Failed to create window");

	return hWnd;
}


//Setting Fullscreen mode
void Ray_Win32Application::ToggleFullscreen()
{	
	mFullscreen = !mFullscreen;
	
	if (mFullscreen) // Switching to Fullscreen.
	{
		// Store the current window dimensions so they can be restored 
		// when switching out of InFullscreen state.
		GetWindowRect(mWindowHandle, &mWindowRect);

		// Set the window style to a borderless window so the client area fills
		// the entire screen.
		UINT windowStyle = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
		SetWindowLongW(mWindowHandle, GWL_STYLE, windowStyle);

		// Query the name of the nearest display device for the window.
		// This is required to set the InFullscreen dimensions of the window
		// when using a multi-monitor setup.
		HMONITOR hMonitor = MonitorFromWindow(mWindowHandle, MONITOR_DEFAULTTONEAREST);
		MONITORINFOEX monitorInfo = {};
		monitorInfo.cbSize = sizeof(MONITORINFOEX);
		GetMonitorInfo(hMonitor, &monitorInfo);

		SetWindowPos(mWindowHandle, HWND_TOP,
			monitorInfo.rcMonitor.left,
			monitorInfo.rcMonitor.top,
			monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left,
			monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top,
			SWP_FRAMECHANGED | SWP_NOACTIVATE);

		ShowWindow(mWindowHandle, SW_MAXIMIZE);
	}
	else
	{
		// Restore all the window decorators.
		SetWindowLong(mWindowHandle, GWL_STYLE, WS_OVERLAPPEDWINDOW);

		SetWindowPos(mWindowHandle, HWND_NOTOPMOST,
			mWindowRect.left,
			mWindowRect.top,
			mWindowRect.right - mWindowRect.left,
			mWindowRect.bottom - mWindowRect.top,
			SWP_FRAMECHANGED | SWP_NOACTIVATE);

		ShowWindow(mWindowHandle, SW_NORMAL);
	}
	
}



i32 Ray_Win32Application::Run(Ray_Sample* pSample, HINSTANCE hInstance, int nCmdShow)
{
	try
	{
		// Windows 10 Creators update adds Per Monitor V2 DPI awareness context.
		// Using this awareness context allows the client area of the window 
		// to achieve 100% scaling while still allowing non-client window content to 
		// be rendered in a DPI sensitive fashion.
		SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

		//Parse command line arguments. Atm the only options we are getting from command line are width/height and wether we want to create a warp device or not
		ParseCommandLineArguments(pSample);

		//NOTE: we need to remove the cached pointer from Win32Application class. A solution based on delegates might solve better the problem of calling/sending messages between the sample/game class and the win32 application class
		mCachedSamplePtr = pSample;

		//Create the render window
		const wchar_t* WindowClassName = L"RaySampleClass";
		
		mWindowHandle = CreateRenderWindow(WindowClassName, hInstance, pSample->GetSampleName(), pSample->GetWidth(), pSample->GetHeight(),WinProc);

		//Init the Ray Sample
		pSample->OnInit();

		//Finally show the window 
		ShowWindow(mWindowHandle, SW_SHOW);

		mAppIsRunning = true;
		while (mAppIsRunning)
		{
			//Get delta time for the current frame 
			const float DeltaFrame = gFrameManager.GetFrameDuration();

			MSG Message = {};
			if (PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
			{
				if (Message.message == WM_QUIT)
				{
					mAppIsRunning = false;
				}

				TranslateMessage(&Message);
				DispatchMessage(&Message);
			}

			//Do any game specific update/render here

			//Update

			//We cap the deltaframe to account for event such as window is dragged from the bar big deltaframe.
			//This is useful because will keep the Tick logic stable.

			pSample->OnUpdate(std::min(DeltaFrame,0.05f));

			//Render
			pSample->OnRender();

		}

		pSample->OnDestroy();

		return 0;

	}
	catch (std::exception &e)
	{
		
		//TODO: Place Log here
		OutputDebugStringA(e.what());

		pSample->OnDestroy();
		return EXIT_FAILURE;
	}

}



LRESULT CALLBACK Ray_Win32Application::WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	//Get a pointer to the hardware renderer
	auto HWRenderer = mCachedSamplePtr->GetHardwareRenderer();

	switch (message)
	{
	case WM_PAINT:
	{
		
	}
	break;

	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
	{
		bool AltButtonDown = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

		switch (wParam)
		{
		case 'V':
			if (HWRenderer)
			{
				HWRenderer->ToggleVSync();
			}
			break;
		case VK_ESCAPE:
			PostQuitMessage(0);
			break;
		case VK_RETURN:

		case VK_F11:
			if (AltButtonDown)
			{				
				ToggleFullscreen();
			}
			break;
		}
	}
	break;
	// The default window procedure will play a system notification sound 
	// when pressing the Alt+Enter keyboard combination if this message is 
	// not handled.
	case WM_SYSCHAR:
		break;

	case WM_QUIT:
		mAppIsRunning = false;
		break;

	case WM_SIZE:
	{
		RECT clientRect = {};
		GetClientRect(mWindowHandle, &clientRect);

		u32 width = clientRect.right - clientRect.left;
		u32 height = clientRect.bottom - clientRect.top;

		if (mCachedSamplePtr)
		{
			mCachedSamplePtr->ResizeWindow(width,height);
		}
		
	}
	break;

	case WM_DESTROY:
		mAppIsRunning = false;
		break;

	default:
		return DefWindowProcW(hWnd, message, wParam, lParam);
	}
		
	return DefWindowProcW(hWnd, message, wParam, lParam);
}
