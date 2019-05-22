
#include "bmpimage.h"

#include <stdio.h>
#include <math.h>

void Ray_BMPSaver::Save(const char* Filename, int32_t Width, int32_t Height, int32_t dpi, float* Data)
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

	//Write out file header and file info header 
	fwrite(BmpFileHeader,1,14,OutFile);
	fwrite(BmpInfoHeader,1,40,OutFile);

	//Copy and convert data before image dump
	for (int32_t i=0;i<Resolution;++i)
	{		 
		float red    = Data[i*3]   * 255.99f;
		float green  = Data[i*3+1] * 255.99f;
		float blue   = Data[i*3+2] * 255.99f;

		//Pixel format must be in the formm BGR not RGB for bmp to work
		uint8_t color[3] = { static_cast<uint8_t>(floor(blue)),static_cast<uint8_t>(floor(green)), static_cast<uint8_t>(floor(red))};

		fwrite(color, 1, 3, OutFile);
	}


	//Done let's close the file 
	fclose(OutFile);

}

float* Ray_BMPSaver::Load(const char* Filename, int32_t& Width, int32_t& Height)
{
	FILE   *InFile = nullptr;

	float* UNORMImageData = nullptr;
	uint8_t* ImageData    = nullptr;

	uint8_t BmpFileHeader[14] = {  };
	uint8_t BmpInfoHeader[40] = {  };

	//Open bmp file
	InFile = fopen(Filename, "rb");

	fread(BmpFileHeader, 1, 14, InFile);
	fread(BmpInfoHeader, 1, 40, InFile);

	//Allocate memory to hold image data
	size_t ImageBytes = (BmpInfoHeader[24] << 24) | (BmpInfoHeader[23] << 16) | (BmpInfoHeader[22] << 8 ) | BmpInfoHeader[21];
	ImageData = new uint8_t[ImageBytes];

	//Read image data from file
	fread(ImageData, 1, ImageBytes, InFile);

	//Done close the file
	fclose(InFile);

	//Read image resolution info
	Width = (BmpInfoHeader[7] << 24) | (BmpInfoHeader[6] << 16) | (BmpInfoHeader[5] << 8) | BmpInfoHeader[4];
	Height = (BmpInfoHeader[11] << 24) | (BmpInfoHeader[10] << 16) | (BmpInfoHeader[9] << 8) | BmpInfoHeader[8];

	const uint32_t Resolution = Width * Height;
	size_t UNORMImagePixels = 3 * Resolution;

	UNORMImageData = new float[UNORMImagePixels];

	const float Temp = 1.0f / 255.99f;
	for (uint32_t i = 0; i < Resolution; ++i)
	{
		uint32_t Offset = i * 3;
		UNORMImageData[Offset]     = static_cast<float>(ImageData[Offset + 2]) * Temp;
		UNORMImageData[Offset + 1] = static_cast<float>(ImageData[Offset + 1]) * Temp;
		UNORMImageData[Offset + 2] = static_cast<float>(ImageData[Offset])     * Temp;
	}


	//done release uint8 memory 
	delete [] ImageData;

	return UNORMImageData;
}