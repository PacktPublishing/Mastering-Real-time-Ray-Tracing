#pragma once

#include "RayPCH.h"
#include "RaySample.h"


class Ray_DX12HelloWorldSample : public Ray_Sample
{
public:

	Ray_DX12HelloWorldSample(u32 Width, u32 Height, std::wstring&& SampleName);

	virtual ~Ray_DX12HelloWorldSample();

	virtual void OnInit() override;

	virtual void OnUpdate(float DeltaFrame) override;

	virtual void OnRender() override;

	virtual void OnDestroy() override;

};