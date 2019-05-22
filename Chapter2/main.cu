#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bmpimage.h"

#include <stdio.h>
#include <iostream>


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

//Here we overload few useful math functions/operators in order to use them in our filter kernel
__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator*(const float3 &a, const float3 &b) {

	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}

__device__ float3 operator/(const float3 &a, const float3 &b) {

	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}

__device__ float3 operator/(const float3 &a, const float b) {

	return make_float3(a.x / b, a.y / b, a.z / b);

}

__device__ float3 operator*(const float3 &a, const float b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}

__device__ int Max(int a, int b)
{
	return a > b ? a : b;
}


__device__ int Min(int a, int b)
{
	return a < b ? a : b;
}

__constant__ const int kFilterRadius = 25;
__constant__ const float FilterWeights[25] = { 0.f,	0.f,	0.000001f,	0.00001f,	0.000078f,	0.000489f,	0.002403f,	0.009245f,	0.027835f,	0.065591f,	0.120978f,	0.174666f,	0.197413f,	0.174666f,	0.120978f,	0.065591f,	0.027835f,	0.009245f,	0.002403f,	0.000489f,	0.000078f,	0.00001f,	0.000001f,	0.f,	0.f };

//the keyword __global__ instructs the CUDA compiler that this function is the entry point of our kernel
__global__ void FilterImageKernel(float* ColorBuffer
	                            , const int Width	
	                            , const int Height
	                            , float* FilteredColorBuffer)
{
	//shared memory 
	__shared__ float3 CachedColors[512];
	
	//Compute global x and t coords
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//checks whether we are inside the color buffer bounds.
    //If not, just return
	if (x >= Width || y >= Height)
	{
		return;
	}

	const int ColorBufferOffset = (x + y * Width) * 3;

	//Start memory transactions here (copy color from global memory to shared fast memory)       
	CachedColors[threadIdx.x].x = ColorBuffer[ColorBufferOffset   ];
	CachedColors[threadIdx.x].y = ColorBuffer[ColorBufferOffset +1];
	CachedColors[threadIdx.x].z = ColorBuffer[ColorBufferOffset +2];

	
	//wait for all the threads in the block to finish their memory transactions before accessing any value store in CachedColors
	__syncthreads();


	//Add filter code here
	int OffsetThreadId = threadIdx.x - kFilterRadius / 2;
	float3 Result = make_float3(0.0f, 0.0f, 0.0f);
	for (int x = 0; x < kFilterRadius; ++x)
	{		
		Result = Result + (CachedColors[Min(Max(0,OffsetThreadId + x),Width-1)] * FilterWeights[x]);
	}
	//write back the filter result in global memory rotating 90° the image (this is a trick to always have coalesced access pattern on global memory read)
	const int RotatedOffset = (y + x * Width) * 3;	
	FilteredColorBuffer[RotatedOffset    ] = Result.x;
	FilteredColorBuffer[RotatedOffset + 1] = Result.y;
	FilteredColorBuffer[RotatedOffset + 2] = Result.z;
}


int main()
{

	float* ColorBuffer = nullptr;
	float* IntermediateResults = nullptr;

	//Here we prepare our computation domain (i.e. thread blocks and threads in a block)

	//Number of threads in a block (experiment with this sizes!). 
	//Suggenstion: make them a multiple of a warp (a warp is 32 threads wide on NVIDIA and 64 threads on AMD)
	int ThreadBlockSizeX = 512;
	int ThreadBlockSizeY = 1;

	//Image Buffer default resolution
	int ImageWidth = 256;
	int ImageHeight = 256;


	//Load an bmp image
	float* ImageData = Ray_BMPSaver::Load("flower.bmp", ImageWidth, ImageHeight);

	//Number of thread blocks
	int NumOfBlockX = ImageWidth / ThreadBlockSizeX;
	int NumOfBlockY = ImageHeight / ThreadBlockSizeY;

	//Let's define the compute dimention domain
	dim3 ThreadBlocks(NumOfBlockX, NumOfBlockY);
	dim3 ThreadsInABlock(ThreadBlockSizeX, ThreadBlockSizeY);	

	//Color buffer size in bytes
	const size_t kColorBufferSize = sizeof(float) * 3 * ImageWidth*ImageHeight;


	//We allocate our color buffer in Unified Memory such that it'll be easy for us to access it on the host as well as on the device
	CHECK_CUDA_ERRORS(cudaMallocManaged(&ColorBuffer, kColorBufferSize));
	CHECK_CUDA_ERRORS(cudaMallocManaged(&IntermediateResults, kColorBufferSize));

	//copy image data from host memory to device memory for processing
	memcpy(ColorBuffer, ImageData, kColorBufferSize);

	//Perform horizontal blur pass
	FilterImageKernel << <ThreadBlocks, ThreadsInABlock >> > (ColorBuffer, ImageWidth, ImageHeight,IntermediateResults);

	//Wait for the GPU to finish before to access results of the previous pass
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	//Perform vertical blur pass
	FilterImageKernel << <ThreadBlocks, ThreadsInABlock >> > (IntermediateResults, ImageWidth, ImageHeight, ColorBuffer);

	//Wait for the GPU to finish before to access results of the final pass
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());


	//Save results stored in ColorBuffer to file (could be a *.ppx or a *.bmp)	

	//We are ready to use the results produced on the GPU
	//Dump Results on a file 
	const int dpi = 72;
	Ray_BMPSaver::Save("Chapter2_CudaResult.bmp", ImageWidth, ImageHeight, dpi, (float*)ColorBuffer);

	//Done! Free up cuda allocated memory
	CHECK_CUDA_ERRORS(cudaFree(IntermediateResults));
	CHECK_CUDA_ERRORS(cudaFree(ColorBuffer));

	return 0;
}

