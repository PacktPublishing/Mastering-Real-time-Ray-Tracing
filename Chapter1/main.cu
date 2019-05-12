
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

	//We access the linear ColorBuffer storing each color component separately (we could have a float3    color buffer for a more compact/cleaner solution)
	int offset = (x + y * N) * 3;

	//Store the results of your computations
	ColorBuffer[offset] = static_cast<float>(x) / static_cast<float>(N);
	ColorBuffer[offset + 1] = static_cast<float>(y) / static_cast<float>(N);
	ColorBuffer[offset + 2] = 0.0f;
}

int main()
{
   	//Color Buffer resolution
	int ScreenWidth    = 512;
	int ScreenHeight   = 512;

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
	dim3 ThreadBlocks(NumOfBlockX,NumOfBlockY);
	dim3 ThreadsInABlock(ThreadBlockSizeX, ThreadBlockSizeY);

	//Color buffer size in bytes
	const size_t kColorBufferSize = sizeof(float) * 3 * ScreenWidth*ScreenHeight;

	//We allocate our color buffer in Unified Memory such that it'll be easy for us to access it on the host as well as on the device
	CHECK_CUDA_ERRORS( cudaMallocManaged(&ColorBuffer,kColorBufferSize) );

	//Launch the kernel that will render the scene
	RenderScene << <ThreadBlocks, ThreadsInABlock >> > (ScreenWidth,ColorBuffer);

	//Wait for the GPU to finish before to access results on the host 
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());


    //Save results stored in ColorBuffer to file (could be a *.ppx or a *.bmp)	
	
	//We are ready to use the results produced on the GPU
	//Dump Results on a file 
	const int dpi = 72;
	Ray_BMPSaver::Save("Chapter1_CudaResult.bmp", ScreenWidth, ScreenHeight, dpi, (float*)ColorBuffer);

	//Done! Free up cuda allocated memory
	CHECK_CUDA_ERRORS(cudaFree(ColorBuffer));

    return 0;
}

