
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/bmpimage.h"
#include "../Utils/vector3.h"

#include <stdio.h>
#include <iostream>

//__constant__ static const float kPI = 3.1415927f;

//__device__ inline float DegToRad(float Deg)
//{
//	return (Deg * kPI / 180.0f);
//}


//Ray tracing data structures

//Simple struct used to collect post hit data (i.e. hit position, normal and t)
struct HitData
{
	/** Ctor */
	__device__ HitData() : mHitPos(0.f,0.f,0.f), mNormal(0.f,1.f,0.f) { }
	
	Vector3 mHitPos;
	
	Vector3 mNormal;
	
	float t = 0.0f;
};

class Camera
{
public:

	__device__ Camera(const Vector3& InEye = Vector3(0.f, 0.f, 0.f), const Vector3& InLookAt = Vector3(0.f, 0.f, 50.f), const Vector3& InUp = Vector3(0.f, 1.f, 0.f), float InFov = 60.f, float InAspectRatio = 1.f) : mEye(InEye), mLookAt(InLookAt)
	{

		const Vector3& Fwd = InLookAt - InEye;
		mW = Fwd.norm();
		mU = mW.cross(InUp);
		mV = mU.cross(mW);

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

	//we could add more accessor (getter/setter) if necessary

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

};


//Simple ray class 
class Ray
{
public:
	
	/** Ctor */
	__device__ Ray(const Vector3& InOrigin = Vector3(0, 0, 0), const Vector3& InDirection = Vector3(0, 0, 1)) : mOrigin(InOrigin), mDirection(InDirection) {}

	/** Copy Ctor */
	__device__ Ray(const Ray& InRay) : mOrigin(InRay.mOrigin), mDirection(InRay.mDirection) { }

	//Method used to compute position at parameter t
	__device__ Vector3 PositionAtT(float t) const
	{
		return mOrigin + mDirection * t;
	}

	Vector3 mOrigin;

	Vector3 mDirection;

	float mTmin;

	float mTmax;

};


//Simple sphere class
class Sphere
{
private:

	/** The center of the sphere */
	Vector3 mCenter;

	/** The radius of the sphere */
	float mRadius;

public:

	/** Ctor */
	__device__ Sphere(const Vector3& InCenter = Vector3(0, 0, 0), float InRadius = 1) : mCenter(InCenter), mRadius(InRadius) {  }

	/** Copy Ctor */
	__device__ Sphere(const Sphere& InSphere) : mCenter(InSphere.mCenter), mRadius(InSphere.mRadius) {  }

	//Compute the ray-sphere intersection using analitic solution
	__device__ bool Intersect(const Ray& InRay, float InTMin, float InTMax, HitData& OutHitData)
	{
		const Vector3& oc = (InRay.mOrigin - mCenter);
		float a = InRay.mDirection.dot(InRay.mDirection);
		float b = oc.dot(InRay.mDirection);
		float c = oc.dot(oc) - mRadius * mRadius;
		float Disc = b * b - a * c;
		float SqrtDisc = sqrt(Disc);
		if (Disc > 0)
		{
			float temp = (-b - SqrtDisc) / a;
			if (temp < InTMax && temp > InTMin)
			{
				OutHitData.t = temp;
				OutHitData.mHitPos = InRay.PositionAtT(temp);
				OutHitData.mNormal = (OutHitData.mHitPos - mCenter) / mRadius;
				return true;
			}
			temp = (-b + SqrtDisc) / a;
			if (temp < InTMax && temp > InTMin)
			{
				OutHitData.t = temp;
				OutHitData.mHitPos = InRay.PositionAtT(temp);
				OutHitData.mNormal = (OutHitData.mHitPos - mCenter) / mRadius;
				return true;
			}

		}
		return false;
	}
};



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

//the keyword __global__ instructs the CUDA compiler that this function is the entry point of our kernel
__global__ void RenderScene(const int N, float* ColorBuffer)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//checks whether we are inside the color buffer bounds.
	//If not, just return
	if (x >= N || y >= N)
	{
		return;
	}

	//Create a simple sphere 10 units away from the world origin
	Sphere sphere(Vector3(0.0f,0.0f,1.0f),1.0f);

	//Prepare two color
	Vector3 Black(0.0f, 0.0f, 0.0f);   //Black background if we miss a primitive
	Vector3 Green(0.0f, 1.0f, 0.0f);  //Red color if we hit a primitive (in our case a sphere, but can be any type of primitive)

	//Create a camera
	Camera camera(Vector3(0.0f,0.0f,-5.0f));

	//Cast a ray in world space from the camera

	//Compute the world space ray direction
	auto WSDir = camera.GetWorldSpaceRayDir(x,y,N,N);

	//Construct a ray in world space that originates from the camera
	Ray WSRay(camera.GetCameraEye(), WSDir);

	//Compute intersection and set a color
	HitData OutHitData;
	Vector3 ColorResult = sphere.Intersect(WSRay,0.001f,FLT_MAX,OutHitData) ? Green : Black;


	//We access the linear ColorBuffer storing each color component separately (we could have a float3    color buffer for a more compact/cleaner solution)
	int offset = (x + y * N) * 3;

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
	RenderScene << <ThreadBlocks, ThreadsInABlock >> > (ScreenWidth, ColorBuffer);

	//Wait for the GPU to finish before to access results on the host 
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());


	//Save results stored in ColorBuffer to file (could be a *.ppx or a *.bmp)	

	//We are ready to use the results produced on the GPU
	//Dump Results on a file 
	const int dpi = 72;
	Ray_BMP_Manager::Save("Chapter3_CudaResult.bmp", ScreenWidth, ScreenHeight, dpi, (float*)ColorBuffer);

	//Done! Free up cuda allocated memory
	CHECK_CUDA_ERRORS(cudaFree(ColorBuffer));

	return 0;
}