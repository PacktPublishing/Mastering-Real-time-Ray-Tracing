#pragma once

#include "ray_utils.h"

#include <math.h>
#include <assert.h>
#include <algorithm>


class Vector3
{
private:

	union
	{
		struct
		{
			float x, y, z;
		};

		float _v[3];
	};

public:
	
	__host__ __device__ Vector3(float InX = 0.0f, float InY = 0.0f, float InZ = 0.0f) : x(InX), y(InY), z(InZ) { }

	__host__ __device__ Vector3(const Vector3& InVec) : x(InVec.x), y(InVec.y), z(InVec.z) { }

	__host__ __device__ Vector3& operator=(const Vector3& InVec)
	{
		x = InVec.x;
		y = InVec.y;
		z = InVec.z;

		return *this;
	}

	__host__ __device__ float operator[](size_t index) const
	{
		assert(index < 3);
		return _v[index];
	}

	__host__ __device__ float X() const
	{
		return  x;
	}

	__host__ __device__ float Y() const
	{
		return  y;
	}

	__host__ __device__ float Z() const
	{
		return  z;
	}

	__host__ __device__ Vector3 operator+(const Vector3& InVec) const
	{
		return Vector3(x + InVec.x, y + InVec.y, z + InVec.z);
	}

	__host__ __device__ Vector3 operator+(float InScalar) const
	{
		return Vector3(x + InScalar, y + InScalar, z + InScalar);
	}

	__host__ __device__ Vector3 operator-(float InScalar) const
	{
		return Vector3(x - InScalar, y - InScalar, z - InScalar);
	}

	__host__ __device__ Vector3& operator+=(const Vector3& InVec)
	{
		x += InVec.x;
		y += InVec.y;
		z += InVec.z;
		return *this;
	}

	__host__ __device__ Vector3 operator-(const Vector3& InVec) const
	{
		return Vector3(x - InVec.x, y - InVec.y, z - InVec.z);
	}

	__host__ __device__ Vector3 operator-() const
	{
		return Vector3(-x, -y, -z);
	}

	__host__ __device__ Vector3& operator-=(const Vector3& InVec)
	{
		x -= InVec.x;
		y -= InVec.y;
		z -= InVec.z;
		return *this;
	}

	__host__ __device__ Vector3 operator*(const Vector3& InVec) const
	{
		return Vector3(x * InVec.x, y * InVec.y, z * InVec.z);
	}

	__host__ __device__ Vector3& operator*=(const Vector3& InVec)
	{
		x *= InVec.x;
		y *= InVec.y;
		z *= InVec.z;
		return *this;
	}

	__host__ __device__ Vector3 operator/(const Vector3& InVec) const
	{
		//assert(InVec.x > 0.0001f && InVec.y > 0.0001f && InVec.z > 0.0001f);
		return Vector3(x / InVec.x, y / InVec.y, z / InVec.z);
	}

	__host__ __device__ Vector3 operator/=(const Vector3& InVec) const
	{
		return Vector3(*this / InVec);
	}

	__host__ __device__ Vector3 operator*(float Scalar) const
	{
		return Vector3(x * Scalar, y * Scalar, z * Scalar);
	}

	__host__ __device__ Vector3& operator*=(float Scalar) 
	{
		x *= Scalar;
		y *= Scalar;
		z *= Scalar;
		return *this;
	}

	__host__ __device__ Vector3 operator/(float Scalar) const
	{
		//assert(Scalar > 0.0001f);
		return Vector3(x / Scalar, y / Scalar, z / Scalar);
	}

	__host__ __device__ Vector3& operator/=(float Scalar)
	{
		x /= Scalar;
		y /= Scalar;
		z /= Scalar;

		return *this;
	}

	__host__ __device__ static Vector3 RhVector(const Vector3& InVec)
	{
		return Vector3(1.0f / InVec.x, 1.0f / InVec.y, 1.0f / InVec.z);

	}

	__host__ __device__ static Vector3 Pow(const Vector3& InVec, float Exp)
	{
		return Vector3(powf(InVec.x, Exp), powf(InVec.y, Exp), powf(InVec.z, Exp));
	}

	__host__ __device__ static Vector3 reflect(const Vector3& I, const Vector3& N)
	{
		return Vector3(I - N*(N.dot(I)*2.0f));
	}


	__host__ __device__ static Vector3 refract(const Vector3& I, const Vector3& N, float IOR)
	{
		float cosi = CLAMP(I.dot(N), -1.0f, 1.0f);
		float etai = 1.0f, etat = IOR;
		Vector3 n = N;
		if (cosi < 0)
		{
			cosi = -cosi;
		}
		else
		{
			float temp = etai;
			etai = etat;
			etat = temp;

			n = -N;
		}
		float eta = etai / etat;
		float k = 1 - eta * eta * (1 - cosi * cosi);
		return k < 0 ? Vector3(0, 0, 0) : (I * eta + n * (eta * cosi - sqrtf(k)));
	}

	__host__ __device__ float dot(const Vector3& InVec) const
	{
		return (x * InVec.x + y * InVec.y + z * InVec.z);
	}

	__host__ __device__ Vector3 cross(const Vector3& InVec)  const
	{
		return Vector3(y*InVec.z - z*InVec.y, (z*InVec.x - x*InVec.z), x*InVec.y - y*InVec.x);
	}

	__host__ __device__ float length2() const
	{
		return (x*x + y*y + z*z);
	}

	__host__ __device__ float length() const
	{
		return sqrtf(length2());
	}

	__host__ __device__ Vector3 norm() const
	{
		float Len = length();
		//assert(Len > 0.0001f);
		return Vector3(*this / Len);
	}

	__host__ __device__ void inPlaceNorm()
	{
		float Len = length();
		//assert(Len > 0.0001f);
		*this = (*this / Len);
	}

};



