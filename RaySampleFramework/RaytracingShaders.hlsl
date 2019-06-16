#ifndef RAYTRACING_SHADERS_HLSL
#define RAYTRACING_SHADERS_HLSL

//#include "RaytracingHlslCompat.h"

struct Viewport
{
	float left;
	float top;
	float right;
	float bottom;
};

struct RayGenCB
{
	Viewport viewport;
	Viewport stencil;
};


// Hold the acceleration data structure (built from the application)
RaytracingAccelerationStructure Scene : register(t0, space0);

// Alias for the triangle intersection attributes
typedef BuiltInTriangleIntersectionAttributes IntersectionAttributes;

// UAV used to store the ray tracing results
RWTexture2D<float4> RenderTarget : register(u0);

// CBuffer passed from the application
ConstantBuffer<RayGenCB> gRayGenCB : register(b0);

struct RayPayload
{
	float4 color;
};


[shader("raygeneration")]
void RayCastingShader()
{
	float2 NormCoords = (float2)DispatchRaysIndex() / (float2)DispatchRaysDimensions();

	// Re-normalize to -1,1 range
	NormCoords.x = NormCoords.x*2.0f - 1.0f;
	NormCoords.y = 1.0f - (NormCoords.y*2.0f - 1.0f);

	// Construct the ray in world space with a perspective projection
	// The projection data and camera data are normally passed from the application through a CBuffer, but 
	// we hardcode them for illustration purposes.
	const float InFov = (60.0f / 180.0f) * 3.1415927f;
	const float HFov = InFov * 0.5f;
	const float AspectRatio = (DispatchRaysDimensions().x / DispatchRaysDimensions().y);
	const float ScaleY = tan(HFov);
	const float ScaleX = ScaleY * AspectRatio;
	// Camera is positioned one unit away from the origin along Z
	const float3 CameraEye = float3(0.0f, 0.0f, -20.f);

	// We assume a fixed camera looking down positive Z 
	// The up axis in our case is Y
	float3 u = float3(1, 0, 0);
	float3 v = float3(0, 1, 0);
	float3 w = float3(0, 0, 1);

	// Here we construct a world space ray starting from the perspective camera position
	float3 rayDir = u * ScaleX + v * ScaleY + w;
	float3 origin = CameraEye;

	// Here we start tracing rays
	// RayDesc is a built in HLSL struct
	RayDesc ray;
	ray.Origin = origin;
	ray.Direction = rayDir;

	// Remember tmin and tmax? We set them here as well to account for certain precision issues
	ray.TMin = 0.001;
	ray.TMax = 100000.0;

	// RayPayload is user defined struct in which we can return the return result of the TraceRay call
	RayPayload payload = { float4(0, 0, 0, 0) };

	// TraceRay is the new HLSL intrinsic that starts the ray tracing process
	TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

	// If we hit an object we return its color
	RenderTarget[DispatchRaysIndex().xy] = payload.color;
}

[shader("closesthit")]
void RayCastingClosestHit(inout RayPayload payload, in IntersectionAttributes attr)
{
	float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
	
	// We return green to be consistent with the previous example
	payload.color = float4(barycentrics, 1.0f);
}

[shader("miss")]
void RayCastingMiss(inout RayPayload payload)
{
    payload.color = float4(0.5f, 0.5f, 1.0f, 1);
}

#endif // RAYTRACING_SHADERS_HLSL