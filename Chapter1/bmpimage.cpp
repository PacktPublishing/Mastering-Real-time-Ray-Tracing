
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