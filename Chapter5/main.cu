#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/bmpimage.h"
#include "../Utils/vector3.h"

#include <stdio.h>
#include <iostream>


// The number of samples/rays we shoot for each pixel for distributed ray tracing
#define SAMPLES 1024

// Do we want to enable Anti Aliasing via jittering?
#define AA_ENABLED 1

//Useful macro to check cuda error code returned from cuda functions
#define CHECK_CUDA_ERRORS(val) Check( (val), #val, __FILE__, __LINE__ )
static void Check(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}


//Ray tracing data structures

//Simple struct used to collect post hit data (i.e. hit position, normal and t)
struct HitData
{
	/** Ctor */
	__device__ HitData() : mHitPos(0.f, 0.f, 0.f), mNormal(0.f, 1.f, 0.f) { }

	Vector3 mHitPos;

	Vector3 mNormal;

	i32 mObjId = -1;

	float t = 0.0f;
};

class Camera
{
public:

	__device__ Camera(const Vector3& InEye = Vector3(0.f, 0.f, 0.f)
		            , const Vector3& InLookAt = Vector3(0.f, 0.f, 50.f)
		            , const Vector3& InUp = Vector3(0.f, 1.f, 0.f)
		            , float InFov = 60.f
		            , float InAspectRatio = 1.f
	                , float InTime0 = 0.0f
	                , float InTime1 = 1.0f) : mEye(InEye), mLookAt(InLookAt),mTime0(InTime0),mTime1(InTime1)
	{

		const Vector3& Fwd = InLookAt - InEye;
		mW = Fwd.norm();
		mU = InUp.cross(mW);
		mV = mW.cross(mU);

		mScaleY = tanf(DegToRad(InFov)*0.5f);
		mScaleX = mScaleY * InAspectRatio;
	}



	~Camera() = default;

	//We calculate the world space ray given the position of the pixel in image space and 
	//the image plane width and height.
	__device__ Vector3 GetWorldSpaceRayDir(float InPx, float InPy, float InWidth, float InHeight)
	{
		float Alpha = ((InPx / InWidth)*2.0f - 1.0f)*mScaleX;
		float Beta = ((1.0f - (InPy / InHeight))*2.0f - 1.0f)*mScaleY;

		Vector3 WSRayDir = mU * Alpha + mV * Beta + mW;

		return WSRayDir;
	}

	__device__ Vector3 GetCameraEye() const { return mEye; }

	__device__ float GetFocalLength() const { return mFocalLength; }

	__device__ float GetApertureSize() const { return mApertureSize; }

private:

	//Convenient member variables used to cache the scale along the x and y axis of the
	//camera space

	float mScaleY = 1.0f;

	float mScaleX = 1.0f;

	/**The camera position */
	Vector3 mEye;
	/**The camera forward vector  */
	Vector3 mW;
	/**The camera side vector*/
	Vector3 mU;
	/**The camera up vector */
	Vector3 mV;
	/**The camera look at */
	Vector3 mLookAt;

	/**Focal length */
	float mFocalLength = 5.25f;

	/**Aperture Size */
	//float mApertureSize = 0.7f;
	float mApertureSize = 0.0f;

	// Motion blur variables

	// Time at which the shutter was open
	float mTime0;

	// Time at which the shutter is closed
	float mTime1;

};


//Simple ray class 
class Ray
{
public:

	/** Ctor */
	__device__ Ray(const Vector3& InOrigin = Vector3(0, 0, 0), const Vector3& InDirection = Vector3(0, 0, 1),float InTime = 0.0f) : mOrigin(InOrigin), mDirection(InDirection),mTime(InTime) {}

	/** Copy Ctor */
	__device__ Ray(const Ray& InRay) : mOrigin(InRay.mOrigin), mDirection(InRay.mDirection) { }

	//Method used to compute position at parameter t
	__device__ Vector3 PositionAtT(float t) const
	{
		return mOrigin + mDirection * t;
	}

	// This ray origin
	Vector3 mOrigin;

	// This ray direction
	Vector3 mDirection;

	// Min t
	float mTmin;

	// Max t
	float mTmax;

	// Added for motion blur
	float mTime = 0.0f;

};


//Simple sphere class
struct Sphere
{
	// We also need for the sphere to move to account for motion blur

	/** The center of the sphere at time 0*/
	Vector3 mCenter0;

	/** The center of the sphere at time 1*/
	Vector3 mCenter1;


    /** Let's give this sphere a color */
	Vector3 mColor;

	/** The radius of the sphere */
	float mRadius;

	/** Time at which the sphere started moving (coincides with camera shutter open) */
	float mTime0 = 0.0f;

	/** Time at which the sphere ended up being (coincides with camera shutter closed) */
	float mTime1 = 1.0f;

	__device__ Vector3 GetCenterAtTime(float Time) const noexcept
	{
		return mCenter0 + (mCenter1 - mCenter0)*((Time - mTime0) / (mTime1 - mTime0));
	}
	
	/** Ctor */
	__device__ Sphere(const Vector3& InCenter0 = Vector3(0, 0, 0), const Vector3& InCenter1 = Vector3(0, 0, 0),const Vector3& InColor = Vector3(1,1,1),float InRadius = 1) : mCenter0(InCenter0), mCenter1(InCenter1), mColor(InColor) ,mRadius(InRadius) {  }

	/** Copy Ctor */
	__device__ Sphere(const Sphere& InSphere) : mCenter0(InSphere.mCenter0), mCenter1(InSphere.mCenter1), mColor(InSphere.mColor),mRadius(InSphere.mRadius) {  }


	/** Get the color of this sphere */
	__device__ Vector3 GetColor() const { return mColor; }

	//Compute the ray-sphere intersection using analitic solution
	__device__ bool Intersect(const Ray& InRay, float InTMin, float InTMax,float& t)
	{
		const Vector3& oc = (InRay.mOrigin - GetCenterAtTime(InRay.mTime));
		float a = InRay.mDirection.dot(InRay.mDirection);
		float b = oc.dot(InRay.mDirection);
		float c = oc.dot(oc) - mRadius * mRadius;
		float Disc = b * b - a * c;	
		float temp = 0.0f;
		if (Disc > 0.0001f)
		{
			float SqrtDisc = sqrt(Disc);
			temp = (-b - SqrtDisc) / a;
			if (temp < InTMax && temp > InTMin)
			{
				t = temp;
				return true;
			}
			temp = (-b + SqrtDisc) / a;
			if (temp < InTMax && temp > InTMin)
			{				
				t = temp;
				return true;
			}

		}
		return false;
	}
};


__device__ static Ray GetDOFRay(const Ray& ray, float ApertureSize, float FocalLength, u32* Seed0, u32* Seed1)
{
	// This is the focal point for a given primary camera ray (Dir is unit length)
	auto P = ray.mOrigin + ray.mDirection * FocalLength;

	// Get two random number in -0.5-0.5 range for each component
	float u1 = GetRandom01(Seed0, Seed1);
	float u2 = GetRandom01(Seed0, Seed1);
	float r1 = 2.0f * M_PI * u1;
	float r2 = u2;
	auto RandVec = Vector3(cosf(r1)*r2, sinf(r1)*r2, 0.0f) * ApertureSize;

	// This is the new ray origin 
	auto NewRayOrigin = ray.mOrigin + RandVec;

	// New ray direction
	auto NewRayDir = (P - NewRayOrigin).norm();

	return Ray(NewRayOrigin, NewRayDir);
}

__device__ static bool GetClosestHit(const Ray& InRay, float InTMin, float InTMax,HitData& OutHitData,Sphere* InSphereList,const u32 InNumSpheres)
{
	float Inf = FLT_MAX;
	float tMin = Inf;
	float t = 0.f;
	for (i32 i=0;i<InNumSpheres;++i)
	{
		if (InSphereList[i].Intersect(InRay, 0.001f, FLT_MAX,t) && t < tMin )
		{
			tMin = t;
			OutHitData.t = t;			
			OutHitData.mHitPos = InRay.PositionAtT(t);
			OutHitData.mObjId = i;			
		}
	}
	return (tMin < Inf);
}

//the keyword __global__ instructs the CUDA compiler that this function is the entry point of our kernel
__global__ void RenderScene(const u32 ScreenWidth,const u32 ScreenHeight, float* ColorBuffer)
{
	u32 x = blockIdx.x*blockDim.x + threadIdx.x;
	u32 y = blockIdx.y*blockDim.y + threadIdx.y;
	 

	//Create a simple sphere list made by two spheres
	const u32 kNumSpheres = 5;
	Sphere SphereList[kNumSpheres] = {
		Sphere(Vector3(0.0f,0.0f,1.0f),Vector3(0.0f,0.5f,1.0f),Vector3(0.0f,1.0f,0.0f)),
		Sphere(Vector3(0.75f,0.0f,3.5f),Vector3(0.75f,0.0f,3.5f),Vector3(1.0f,0.0f,0.0f)),
		Sphere(Vector3(1.5f,0.0f,6.0f),Vector3(1.5f,0.35f,6.0f),Vector3(1.0f,1.0f,0.0f)),
		Sphere(Vector3(-0.5f,0.0f,0.0f),Vector3(-0.5f,0.0f,0.0f),Vector3(0.0f,0.0f,1.0f)),
		Sphere(Vector3(-1.0f,0.0f,-1.0f),Vector3(-1.0f,0.0f,-1.0f),Vector3(1.0f,0.0f,1.0f))
	};


	//Prepare two color
	Vector3 Green(0.0f, 1.0f, 0.0f);  //Red color if we hit a primitive (in our case a sphere, but can be any type of primitive)

	//Create a camera
	Camera camera(Vector3(0.0f, 2.0f, -5.0f));

	//Cast a ray in world space from the camera

	u32 Seed0 = x;
	u32 Seed1 = y;	

	// This is the resultant color 
	Vector3 ColorResult;

	// Inv of the number of samples
	const float InvSamples = 1.0f / ((float)SAMPLES);

	// Get parameters needed for DOF computation
	float FocalLength = camera.GetFocalLength();
	float ApertureSize = camera.GetApertureSize();

	// We use this to create a gradient for the background color. It will be a bit prettier to look at and it will give a better contrast with objects in the foreground
	float GradientInterp = (static_cast<float>(y) / static_cast<float>(ScreenHeight));
	Vector3 BkgColor = Lerp(Vector3(0.671f,0.875f,0.973f), Vector3(0.992f,0.941f,0.918f),GradientInterp);

	// For each sample of a given pixel
	// We implement distributed ray casting
	// What we obtain at the end of this process is antialiasing 
	for (u32 i = 0; i < SAMPLES; ++i)
	{
		// Pick a random sample in 0-1 range
#if AA_ENABLED
		float rx = GetRandom01(&Seed0, &Seed1);
		float ry = GetRandom01(&Seed0, &Seed1);

		//Compute the world space ray direction for each sample and then average the results
		auto WSDir = camera.GetWorldSpaceRayDir(((float)x) + rx, ((float)y) + ry , ScreenWidth, ScreenHeight);
#else
		auto WSDir = camera.GetWorldSpaceRayDir(((float)x), ((float)y), ScreenWidth, ScreenHeight);
#endif

		//Construct a ray in world space that originates from the camera
		Ray WSRay(camera.GetCameraEye(), WSDir);

		// Get Random time interval between 0-1
		WSRay.mTime = GetRandom01(&Seed0, &Seed1);

		//Ray DOFRay = GetDOFRay(WSRay, ApertureSize, FocalLength, &Seed0, &Seed1);

		//Compute intersection and set a color
		HitData OutHitData;

		// Get the closest hit
		//bool Hit = GetClosestHit(DOFRay, 0.001f, FLT_MAX, OutHitData,SphereList,kNumSpheres);
		bool Hit = GetClosestHit(WSRay, 0.001f, FLT_MAX, OutHitData, SphereList, kNumSpheres);
		
		// Return the color for a given sample and accumulate the result
		ColorResult += (Hit ? SphereList[OutHitData.mObjId].mColor : BkgColor);
	}

	// Average the results
	ColorResult *= InvSamples;

	//We access the linear ColorBuffer storing each color component separately (we could have a float3    color buffer for a more compact/cleaner solution)
	int offset = (x + (ScreenHeight - y - 1) * ScreenWidth) * 3;

	//Store the results of your computations
	ColorBuffer[offset] = ColorResult.X();
	ColorBuffer[offset + 1] = ColorResult.Y();
	ColorBuffer[offset + 2] = ColorResult.Z();
}

int main()
{
	//Color Buffer resolution
	int ScreenWidth = 512;
	int ScreenHeight = 512;

	float* ColorBuffer = nullptr;

	//Here we prepare our computation domain (i.e. thread blocks and threads in a block)

	//Number of threads in a block (experiment with this sizes!). 
	//Suggenstion: make them a multiple of a warp (a warp is 32 threads wide on NVIDIA and 64 threads on AMD)
	int ThreadBlockSizeX = 8;
	int ThreadBlockSizeY = 8;

	//Number of thread blocks
	int NumOfBlockX = ScreenWidth / ThreadBlockSizeX + 1;
	int NumOfBlockY = ScreenHeight / ThreadBlockSizeY + 1;

	//Let's define the compute dimention domain
	dim3 ThreadBlocks(NumOfBlockX, NumOfBlockY);
	dim3 ThreadsInABlock(ThreadBlockSizeX, ThreadBlockSizeY);

	//Color buffer size in bytes
	const size_t kColorBufferSize = sizeof(float) * 3 * ScreenWidth*ScreenHeight;

	//We allocate our color buffer in Unified Memory such that it'll be easy for us to access it on the host as well as on the device
	CHECK_CUDA_ERRORS(cudaMallocManaged(&ColorBuffer, kColorBufferSize));

	//Launch the kernel that will render the scene
	RenderScene << <ThreadBlocks, ThreadsInABlock >> > (ScreenWidth, ScreenHeight, ColorBuffer);

	//Wait for the GPU to finish before to access results on the host 
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());


	//Save results stored in ColorBuffer to file (could be a *.ppx or a *.bmp)	

	//We are ready to use the results produced on the GPU
	//Dump Results on a file 
	const int dpi = 72;
	Ray_BMP_Manager::Save("Chapter5_CudaResult.bmp", ScreenWidth, ScreenHeight, dpi, (float*)ColorBuffer);

	//Done! Free up cuda allocated memory
	CHECK_CUDA_ERRORS(cudaFree(ColorBuffer));

	return 0;
}
