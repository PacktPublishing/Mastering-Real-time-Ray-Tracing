/*

Freeglut Copyright
------------------

Freeglut code without an explicit copyright is covered by the following
copyright :

Copyright(c) 1999 - 2000 Pawel W.Olszta.All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies or substantial portions of the Software.

The above  copyright notice  and this permission notice  shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE  IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING  BUT  NOT LIMITED  TO THE WARRANTIES  OF MERCHANTABILITY,
FITNESS  FOR  A PARTICULAR PURPOSE  AND NONINFRINGEMENT.IN  NO EVENT  SHALL
PAWEL W.OLSZTA BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,d WHETHER
IN  AN ACTION  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF  OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of Pawel W.Olszta shall not be
used  in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from Pawel W.Olszta.

*/


#include "pch.h"

using namespace std;

// Intial window width
static int gWindowWidth = 1300;
static int gWindowHeight = 512;

// Number of samples we want to add to our distributions
static const int kNumberOfSamples = 512;

// What is the scale for our graph 
static const int kGraphScale = 256;

// The code for R1 and R2 sequences presented here took inspiration from this article http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

static const float kAlpha = 0.618034f;

static const float kPhi2D = 1.32471795724474602596f;
static const float kAlpha2Dx = (1.0f / kPhi2D);
static const float kAlpha2Dy = (1.0f / (kPhi2D*kPhi2D));

// Quasi-random 1D irrational sequence of the family of sequences of Kronecker recurrence methods (R1-sequence)
void RSequence(uint32_t N)
{
	float s0 = 0.0f;
	for (uint32_t i = 1; i <= N; ++i)
	{
		printf("%f ", s0 + fmod(kAlpha*static_cast<float>(i), 1.0f));
	}
	printf("\n");
}


// Quasi-random 2D irrational sequence of the family of sequences of Kronecker recurrence methods (R2-sequence)
void R2Sequence(uint32_t N, uint32_t Scale)
{
	float s0 = 0.0f;
	glBegin(GL_POINTS);
		for (uint32_t i = 1; i <= N; ++i)
		{
			glVertex2f(s0 + fmod(kAlpha2Dx*static_cast<float>(i), 1.0f)*Scale, s0 + fmod(kAlpha2Dy*static_cast<float>(i), 1.0f)*Scale);
		}
	glEnd();
}

// Pure random sequence
void RandomSequence(uint32_t N, uint32_t Scale)
{
	// Prepare random gens
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<float> distribution;

	glBegin(GL_POINTS);
		for (uint32_t i = 1; i <= N; ++i)
		{
			float u = distribution(generator);
			float v = distribution(generator);
			glVertex2f(u * Scale,v * Scale);
		}
	glEnd();
}


// Quasi-random 2D halton sequence

//FUNCTION(index, base)
//BEGIN
//	result = 0;
//	f = 1 / base;
//	i = index;
//	WHILE(i > 0)
//		BEGIN
//			result = result + f * (i % base);
//			i = FLOOR(i / base);
//			f = f / base;
//		END
//	RETURN result;
//END

// Get the halton value given the index of the sample we wanto to generate and a base parameter (must be a prime number)
float GetHalton(uint32_t Index, float Base)
{
	float f = 1.f / Base;
	float Result = 0.f;
	uint32_t i = Index;
	while (i > 0)
	{
		Result += f*fmodf((float)i, Base);			
		i = (uint32_t)floor(i / Base);
		f /= Base;
	}
	return Result;
}

// We actually generate the halton sequence in 2D for base value of 2 and 3 (they are coprime numbers)
void HaltonSequence2D(uint32_t N, uint32_t Scale)
{
	// Plot Halton(2,3) sequence
	glBegin(GL_POINTS);
		for (uint32_t i = 1; i <= N; ++i)
		{
			glVertex2f(GetHalton(i, 2 )*Scale, GetHalton(i, 3)*Scale);
		}
	glEnd();
}


// Quasi-random 2D Hammersley 2D sequence

// Radical inverse taken from Hacker's Delight [warren01] provides us a a simple way to reverse the bits in a given 32bit integer.
// This implementation comes from http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html#warren01

float RadicalInverse_VdC(uint32_t bits) 
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return (float)(float(bits) * 2.3283064365386963e-10); // / 0x100000000
}

void HammersleySequence2D(uint32_t N, uint32_t Scale)
{
	// Plot Hammersley sequence
	const float InvN = 1.0f / float(N);
	glBegin(GL_POINTS);
		for (uint32_t i = 1; i <= N; ++i)
		{
			glVertex2f(float(i) * InvN * Scale, RadicalInverse_VdC(i) * Scale);
		}
	glEnd();
}


// Helper functions to help drawing graph and sequences

// Draw the graph axis at a given 2D position on-screen (x,y)
void DrawGraphAxisAtPosition(float x, float y)
{
	glPushMatrix();
	glLoadIdentity();

	glTranslatef(x, y, 0);

	glColor3f(0, 0, 0);

	glLineWidth(2.f);
	glBegin(GL_LINES);

		// Vertical axis
		glVertex2f(0, 0);
		glVertex2f(0, kGraphScale);

		// Horizontal axis
		glVertex2f(0, 0);
		glVertex2f(kGraphScale, 0);

	glEnd();

	glPopMatrix();
}


// Draw a given sequence at a given position
void PlotSequenceAtPostion(float x, float y,float* RGB, uint32_t N, uint32_t Scale, void(*SequenceCallback) (uint32_t N, uint32_t Scale))
{
	glPointSize(4.f);
	glPushMatrix();
	glLoadIdentity();

	glColor3f(RGB[0], RGB[1], RGB[2]);

	glTranslatef(x, y, 0);

	SequenceCallback(N,Scale);

	glPopMatrix();
}


void Display(void)
{
	// Clear the buffer 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Border and Horizontal offset 
	const auto kBorder = 32.f;
	const auto kHOffset = kGraphScale + kBorder;
	const float kHeight = (float)(gWindowHeight / 2 - 128);

	// R2 Sequence 
	float Color[3] = {0,0,1};
	DrawGraphAxisAtPosition(kBorder, kHeight);

	PlotSequenceAtPostion(kBorder, kHeight, Color, kNumberOfSamples, kGraphScale, R2Sequence);

	// Draw the pure random sequence 
	DrawGraphAxisAtPosition(kBorder + kHOffset, kHeight);

	Color[0] = 1;
	Color[1] = 0;
	Color[2] = 0;
	PlotSequenceAtPostion(kBorder + kHOffset, kHeight, Color, kNumberOfSamples, kGraphScale, RandomSequence);

	// Draw Halton sequence 
	DrawGraphAxisAtPosition(kBorder + kHOffset *2, kHeight);

	Color[0] = 0;
	Color[1] = 1;
	Color[2] = 0;
	PlotSequenceAtPostion(kBorder + kHOffset *2, kHeight, Color, kNumberOfSamples, kGraphScale, HaltonSequence2D);

	// Draw Hammersley sequence
	DrawGraphAxisAtPosition(kBorder + kHOffset * 3, kHeight);

	Color[0] = 1;
	Color[1] = 0;
	Color[2] = 1;
	PlotSequenceAtPostion(kBorder + kHOffset * 3, kHeight, Color, kNumberOfSamples, kGraphScale, HammersleySequence2D);


	// Swap to visualize the contento of the backbuffer
	glutSwapBuffers();
}


void Reshape(int w, int h)
{
	// Update the 
	gWindowHeight = h;
	gWindowWidth = w;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	
	// Set the correct perspective.
	gluOrtho2D(0, w, 0, h);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);	

}


int main(int argc, char **argv)
{
	// Init glut 
	glutInit(&argc, argv);
	
	// Init display mode
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	
	// Init window size
	glutInitWindowSize(gWindowWidth, gWindowHeight);
	
	// Actually create the window
	glutCreateWindow("Quasi-Random Sequences");
	
	// Set the rendering callback function
	glutDisplayFunc(Display);
	
	// Set the resize callback
	glutReshapeFunc(Reshape);
	
	// Clear the screen to white
	glClearColor(1, 1, 1, 1);

	// Enter the glut main loop
	glutMainLoop();

	return 0;             
}

