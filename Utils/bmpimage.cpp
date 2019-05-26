
#include "bmpimage.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

void Ray_BMP_Manager::Save(const char* Filename, int32_t Width, int32_t Height, int32_t dpi, float* Data)
{
	FILE   *OutFile    = nullptr;
	int32_t Resolution = Width*Height;
	int32_t s          = 4*Resolution;
	int32_t FileSize   = 54 + s;

	double Factor      = 39.375;
	int32_t m          = static_cast<int32_t>(Factor);

	int32_t ppm        = dpi*m;


	//BMP File header 
	uint8_t BmpFileHeader[14] = {'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0};
	uint8_t BmpInfoHeader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,24,0};	

	BmpFileHeader[2] = static_cast<uint8_t>(FileSize);
	BmpFileHeader[3] = static_cast<uint8_t>(FileSize>>8);
	BmpFileHeader[4] = static_cast<uint8_t>(FileSize>>16);
	BmpFileHeader[5] = static_cast<uint8_t>(FileSize>>24);

	BmpInfoHeader[4] = static_cast<uint8_t>(Width);
	BmpInfoHeader[5] = static_cast<uint8_t>(Width >> 8);
	BmpInfoHeader[6] = static_cast<uint8_t>(Width >> 16);
	BmpInfoHeader[7] = static_cast<uint8_t>(Width >> 24);

	BmpInfoHeader[8] = static_cast<uint8_t>(Height);
	BmpInfoHeader[9] = static_cast<uint8_t>(Height >> 8);
	BmpInfoHeader[10] = static_cast<uint8_t>(Height >> 16);
	BmpInfoHeader[11] = static_cast<uint8_t>(Height >> 24);


	BmpInfoHeader[21] = static_cast<uint8_t>(s);
	BmpInfoHeader[22] = static_cast<uint8_t>(s >> 8);
	BmpInfoHeader[23] = static_cast<uint8_t>(s >> 16);
	BmpInfoHeader[24] = static_cast<uint8_t>(s >> 24);

	BmpInfoHeader[25] = static_cast<uint8_t>(ppm);
	BmpInfoHeader[26] = static_cast<uint8_t>(ppm >> 8);
	BmpInfoHeader[27] = static_cast<uint8_t>(ppm >> 16);
	BmpInfoHeader[28] = static_cast<uint8_t>(ppm >> 24);

	BmpInfoHeader[29] = static_cast<uint8_t>(ppm);
	BmpInfoHeader[30] = static_cast<uint8_t>(ppm >> 8);
	BmpInfoHeader[31] = static_cast<uint8_t>(ppm >> 16);
	BmpInfoHeader[32] = static_cast<uint8_t>(ppm >> 24);


	//Save out results on a bmp file
	OutFile = fopen(Filename,"wb");
	if (OutFile == nullptr)
	{
		printf("Failed opening file %s for write\n", Filename);
		return;
	}

	//Write out file header and file info header 
	fwrite(BmpFileHeader,1,14,OutFile);
	fwrite(BmpInfoHeader,1,40,OutFile);

	//Allocate a temp buffer to hold the image color data that then we will dump on file with just one fwrite
	size_t Pixels = Resolution * 3;
	
	uint8_t*  TempImageBuffer = new uint8_t[Pixels];
	
	memset(TempImageBuffer,0,Pixels);

	//Copy and convert data before image dump	
	for (int32_t y=Height-1;y >= 0;--y)
		for(int32_t x = Width-1; x >= 0; --x)
	{		 
		const uint32_t Offset = (x+y*Width) * 3;

		float Red    = Data[Offset   ] * 255.99f;
		float Green  = Data[Offset +1] * 255.99f;
		float Blue   = Data[Offset +2] * 255.99f;

		TempImageBuffer[Offset + 2] = static_cast<uint8_t>(floor(Red));
		TempImageBuffer[Offset + 1] = static_cast<uint8_t>(floor(Green));
		TempImageBuffer[Offset    ] = static_cast<uint8_t>(floor(Blue));;
	}

	//Dump image data on file
	fwrite(TempImageBuffer, 1, Pixels, OutFile);

	//done release the previosly allocated memory 
	delete[] TempImageBuffer;

	//Done let's close the file 
	auto Result = fclose(OutFile);
	if (Result == EOF)
	{
		printf("Failed closing file %s opened for write\n", Filename);
	}	
}

float* Ray_BMP_Manager::Load(const char* Filename, int32_t& Width, int32_t& Height)
{
	FILE   *InFile = nullptr;

	float* UNORMImageData = nullptr;
	uint8_t* ImageData    = nullptr;

	uint8_t BmpFileHeader[14] = {  };
	uint8_t BmpInfoHeader[40] = {  };

	//Open bmp file
	InFile = fopen(Filename, "rb");
	if (InFile == nullptr)
	{
		printf("Failed opening file %s for read\n", Filename);
		return nullptr;
	}

	//Read in bmp file info header and bmp file header 
	fread(BmpFileHeader, 1, 14, InFile);
	fread(BmpInfoHeader, 1, 40, InFile);

	//Read image resolution info
	Width = (BmpInfoHeader[7] << 24) | (BmpInfoHeader[6] << 16) | (BmpInfoHeader[5] << 8) | BmpInfoHeader[4];
	Height = (BmpInfoHeader[11] << 24) | (BmpInfoHeader[10] << 16) | (BmpInfoHeader[9] << 8) | BmpInfoHeader[8];

	const uint32_t Resolution = Width * Height;
	size_t UNORMImagePixels = 3 * Resolution;

	//Allocate memory to hold image data	
	ImageData = new uint8_t[UNORMImagePixels];

	//Clean-up the whole memory to 0
	memset(ImageData, 0, sizeof(uint8_t) * UNORMImagePixels);

	//Read image data from file
	fread(ImageData, 1, UNORMImagePixels, InFile);

	//Done close the file	
	auto Result = fclose(InFile);
	if (Result == EOF)
	{
		printf("Failed closing file %s opened for read\n", Filename);
	}

    //Allocate the buffer that will hold the the fp normalized colors of the given image			
	UNORMImageData = new float[UNORMImagePixels];

	//Clean-up the whole memory to 0
	memset(UNORMImageData, 0, sizeof(float) * UNORMImagePixels);

	//Store image data and then return the pointer 
	const float Temp = 1.0f / 255.99f;
	for (int32_t y = Height - 1; y >= 0; --y)
		for (int32_t x = Width-1; x >= 0; --x)
	{
		const uint32_t Offset = (x + y * Width) * 3;
		UNORMImageData[Offset]     = static_cast<float>(ImageData[Offset + 2]) * Temp;
		UNORMImageData[Offset + 1] = static_cast<float>(ImageData[Offset + 1]) * Temp;
		UNORMImageData[Offset + 2] = static_cast<float>(ImageData[Offset])     * Temp;
	}


	//done release uint8 memory 
	delete [] ImageData;

	return UNORMImageData;
}