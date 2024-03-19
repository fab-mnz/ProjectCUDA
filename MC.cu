/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}


__global__ void init_curand_state_k(curandState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}


__global__ void MC_k(float S_0, float I_0, float r, float sigma, float sqrt_dt, int N_cuts, int M, curandState *state, float *d_samples){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

	float2 G;
	float S = S_0;
	float I = I_0;

	float R1s, R2s;


	for (int n_dt = 0; n_dt < N_cuts; n_dt++) {
		G_out = curand_normal(&localState);
		S *= (1+r*sqrt_dt*sqrt_dt+sigma*sqrt_dt*G_out); // Euler 
		I = (n_dt+1.0f * I + S)/(n_dt+2.0f);

		float R1 = 0
		float R2 = 0
		for (int j = 0; j < M; j++) {
			float S_in = S;
			float I_in = I;
			for (int n_dt_internal = 0; n_dt_internal < N_cuts - (n_dt+1); n_dt_internal++) {
				G_in = curand_normal(&localState);
				S_in *= (1+r*sqrt_dt*sqrt_dt+sigma*sqrt_dt*G_in); 
				I_in = (n_dt_internal+1.0f * I_in + S_in)/(n_dt_internal+2.0f);
			}
			R1 += expf(-r * sqrt_dt * sqrt_dt * N_cuts) * fmaxf(0.0f, S_in - I_in) / M;
			R2 += R1 * R1 * M;
		}
		d_samples[idx] = n_dt * sqrt_dt * sqrt_dt;
		d_samples[idx+1] = S;
		d_samples[idx+2] = I;
		d_samples[idx+3] = R1;
		d_samples[idx+4] = R2;
	}

	/* Copy state back to global memory */
	//state[idx] = localState;
}

int main(void) {

	int NTPB = 1024;
	int NB = 1024;
	int n = NB * NTPB;

	float T = 1.0f;
	float S_0 = 100.0f;
	float I_0 = 100.0f

	int N_sim = 1000000;
	int N_cuts = 100;

	int M = 100; // Number of Monte-Carlo runs for each cut on each simulation

	float sigma = 0.2f;
	float r = 0.1f;
	float sqrt_dt = sqrtf(T/N);

	float* d_samples;
	cudaMallocManaged(&d_samples, 5*N_sim*N_cuts*sizeof(float));
	cudaMemset(d_samples, 0, 5*N_sim*N_cuts*sizeof(float));

	curandState* states;
	cudaMalloc(&states, n*sizeof(curandState));
	init_curand_state_k<<<NB, NTPB>>>(states);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	MC_k<<<NB, NTPB, 2*NTPB*sizeof(float)>>>(S_0, r, sigma, dt, K, N, states, sum, n);

	cudaEventRecord(stop, 0);					// GPU timer instructions
	cudaEventSynchronize(stop);					// GPU timer instructions
	cudaEventElapsedTime(&Tim, start, stop);	// GPU timer instructions
	cudaEventDestroy(start);					// GPU timer instructions
	cudaEventDestroy(stop);						// GPU timer instructions


	printf("The estimated price is equal to %f\n", sum[0]);
	printf("error associated to a confidence interval of 95%% = %f\n", 1.96 * sqrt((double)(sum[1] - (sum[0] * sum[0])))/sqrt((double)n));
	printf("The true price %f\n", S_0 * NP((r + 0.5 * sigma * sigma)/sigma) - K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));
	printf("Execution time %f ms\n", Tim);

	cudaFree(sum);
	cudaFree(states);

	return 0;
}
