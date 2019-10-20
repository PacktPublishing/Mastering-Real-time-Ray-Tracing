

#include <iostream>
#include <random>

// This is the real value for  PI
static const double kREAL_PI = 3.1415927;

// N is the number of samples we've chosen to throw

// Here we consider a quarter of unit circle to compute the area

// We use the hit or miss approach
double HitOrMissMonteCarlo_EstimatePI(int N)
{
	// Prepare random gens
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<> distribution;

	// Total number of hits
	int hits = 0;

	// Start throwing samplesd
	for (int i = 0; i < N; ++i)
	{
		double x = distribution(generator);
		double y = distribution(generator);

		if (sqrt(x * x + y * y) <= 1.0)
		{
			++hits;
		}
	}
	
	// Now we just average the total number of hits by the total number of thrown samples
	return (static_cast<double>(hits) / static_cast<double>(N)) * 4.0;
}


// Here we consider a quarter of unit circle to compute the are

// Use the sample mean approach

// Equation of a unit circle
double f(double x)
{
	return sqrt(1.0 - x*x);
}

// Here the the integration interval [a,b] is [0,1], therefore we evaluate the sample mean considering that (b-a) = (1-0) = 1
// In fact we use the basic Monte Carlo estimator (i.e. the one that uses a uniform probability density function pdf=1/(b-a))
double SampleMeanMonteCarlo_EstimatePI(int N)
{
	// Prepare random gens
	std::random_device device;
    std::mt19937 generator(device());
	std::uniform_real_distribution<> distribution;

	// Integration interval
	double a = 0.0;
	double b = 1.0;

	// Prepare to sum uniform randomly generated samples for f(x)
	double sum = 0.0;
	for (int i = 0; i < N; ++i)
	{
		// Generate a uniformly distributed samples in [0,1)
		double x = distribution(generator);
	
		// Sum the x sample
		sum += f(x);
	}

	// Estimate after N samples
	// Finally average them and multiply by 4 (remember we are considering the integration interval [0,1] for our circle)
	double En = (sum * (b - a) / static_cast<double>(N)) * 4.0;

	return En;
}


int main()
{
	// Enter the number of samples 
	std::cout << "Enter the number of samples to estimate PI: ";

	// Enter the number of samples
	int NumberOfSamples;
	std::cin >> NumberOfSamples;

	std::cout << std::endl;

	// Estimate PI with hit or miss Monte Carlo
	double PI = HitOrMissMonteCarlo_EstimatePI(NumberOfSamples);

	// Estimate PI with the sample mean Monte Carlo estimator
	//double PI = SampleMeanMonteCarlo_EstimatePI(NumberOfSamples);
	
	// Print the result on console
	std::cout << "Estimate of PI for " << NumberOfSamples << " samples: " << PI << '\n';
	std::cout << "Estimate error " << std::abs(PI-kREAL_PI) << "\n";

	return 0;
}
