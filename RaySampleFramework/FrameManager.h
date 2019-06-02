#pragma once

#include <chrono>

//This class uses chrono library to provide frame timing management features
class FrameManager
{

public:
	
	using SteadyClock = std::chrono::steady_clock;

	FrameManager() { mLast = SteadyClock::now(); }

	FrameManager(const FrameManager& rhs) = delete;

	FrameManager& operator=(const FrameManager& rhs) = delete;

	float GetFrameDuration()
	{
		const auto Old = mLast;

		mLast = SteadyClock::now();

		const std::chrono::duration<float> DeltaFrame = mLast - Old;

		return DeltaFrame.count();
	}

private:

	SteadyClock::time_point mLast;

};
