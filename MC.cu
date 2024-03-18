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
	else {/* x < 0 */
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


__global__ void MC_k(float S_0, float r, float sigma, float dt, float K, 
					int N, curandState *state, float *sum, int n){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];
	float2 G;
	float S = S_0;
	extern __shared__ float A[];

	float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

	for (int i = 0; i < N; i++) {
		G = curand_normal2(&localState);
		S *= (1+r*dt*dt+sigma*dt*G.x);
	}
	R1s[threadIdx.x] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K)/ n;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * n;

	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}
	/* Copy state back to global memory */
	//state[idx] = localState;
}

int main(void) {

	int NTPB = 1024;
	int NB = 1024;
	int n = NB * NTPB;
	float T = 1.0f;
	float S_0 = 50.0f;
	float K = S_0;
	float sigma = 0.2f;
	float r = 0.1f;
	int N = 100;
	float dt = sqrtf(T/N);
	float *sum;
	cudaMallocManaged(&sum, 2*sizeof(float));
	cudaMemset(sum, 0, 2*sizeof(float));


	curandState* states;
	cudaMalloc(&states, n*sizeof(curandState));
	init_curand_state_k<<<NB, NTPB>>>(states);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	MC_k<<<NB, NTPB, 2*NTPB*sizeof(float)>>>(S_0, r, sigma, dt, K, 
											N, states, sum, n);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions


	printf("The estimated price is equal to %f\n", sum[0]);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(sum[1] - (sum[0] * sum[0])))/sqrt((double)n));
	printf("The true price %f\n", S_0 * NP((r + 0.5 * sigma * sigma)/sigma) -
									K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));
	printf("Execution time %f ms\n", Tim);


	cudaFree(sum);
	cudaFree(states);

	return 0;
}