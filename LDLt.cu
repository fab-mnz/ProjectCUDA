/**********************************************************************
Code associated to the paper:

Resolution of a large number of small random symmetric 
linear systems in single precision arithmetic on GPUs

by: 

Lokman A. Abbas-Turki and Stef Graillat

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/

#include "cadna.h"
#include "cadna_gpu.cu"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "LDLt.h"

/**********************************************************************

Only factorization

**********************************************************************/
// Optimized for matrices smaller than 4x4: one thread per system
__global__ void LDLt_fact_k(float_gpu_st *a, int n)
{
	// Thread identifier in the grid
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, j, k, n2, n2p1;

	n2 = (n*n+n)/2;
	n2p1 = n2 + n;

	// Copy the lower triangular part from global to shared memory
	for (i=0; i<n; i++){
		for (j=0; j<=i; j++){
			sA[threadIdx.x*n2p1+i*(i+1)/2+j] = a[tidx*n*n+i*n+j];
		}
	}

	// Perform the LDLt factorization
	for(i=0; i<n; i++){
		for(j=0; j<i; j++){
			//Mat[i*Dim+j] /= Mat[j*Dim+j];
			sA[threadIdx.x*n2p1+i*(i+1)/2+j] = sA[threadIdx.x*n2p1+i*(i+1)/2+j]/sA[threadIdx.x*n2p1+j*(j+1)/2+j];
			for(k=0; k<j; k++){
				//Mat[i*Dim+j] -= Mat[k*Dim+k]*Mat[i*Dim+k]*Mat[j*Dim+k]/Mat[j*Dim+j];
				sA[threadIdx.x*n2p1+i*(i+1)/2+j] = sA[threadIdx.x*n2p1+i*(i+1)/2+j] - sA[threadIdx.x*n2p1+k*(k+1)/2+k]*
													sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
													sA[threadIdx.x*n2p1+j*(j+1)/2+k]/
													sA[threadIdx.x*n2p1+j*(j+1)/2+j];
			}
		}
		for(k=0; k<i; k++){
			//Mat[i*Dim+i] -= Mat[k*Dim+k]*Mat[i*Dim+k]*Mat[i*Dim+k];
			sA[threadIdx.x*n2p1+i*(i+1)/2+i] = sA[threadIdx.x*n2p1+i*(i+1)/2+i] - sA[threadIdx.x*n2p1+k*(k+1)/2+k]*
												sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
												sA[threadIdx.x*n2p1+i*(i+1)/2+k]; 
		}
	}

	// Copy the factorization from shared to global memory
	for (i=0; i<n; i++){
		for (j=0; j<=i; j++){
			a[tidx*n*n+i*n+j] = sA[threadIdx.x*n2p1+i*(i+1)/2+j];
		}
	}
}


// Optimized for matrices bigger than 64x64: x64 number of threads
__global__ void LDLt_fact_max_k(float_gpu_st *a, int n)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n; 
    // Identifies the data concerned by the computations
	int Qt = (threadIdx.x-tidx)/n; 
    // The global memory access index
    int gb_index_x = Qt + blockIdx.x*(blockDim.x/n);
	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, k, n2, nt;

    n2 = (n*n+n)/2;
	nt = Qt*(n2 + n);
	
	// Copy the upper triangular part from global to shared memory
	for (i=n; i>0; i--){
		if (tidx<i){
			sA[nt+n2-i*(i+1)/2+tidx] = a[gb_index_x*n*n+(n-i)*(n+1)+tidx];
		}
	}

	__syncthreads();

	// Perform the LDLt factorization
	for(i=n; i>0; i--){
		if(tidx==0){
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2] = sA[nt+n2-i*(i+1)/2] - sA[nt+n2-k*(k+1)/2]*
									   sA[nt+n2-k*(k+1)/2+k-i]*
									   sA[nt+n2-k*(k+1)/2+k-i];
			}
		}
		__syncthreads();
		if(tidx<i-1){
			sA[nt+n2-i*(i+1)/2+tidx+1] = sA[nt+n2-i*(i+1)/2+tidx+1]/sA[nt+n2-i*(i+1)/2];
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2+tidx+1] = sA[nt+n2-i*(i+1)/2+tidx+1]-sA[nt+n2-k*(k+1)/2]*
											  sA[nt+n2-k*(k+1)/2+k-i]*
											  sA[nt+n2-k*(k+1)/2+tidx+1]/
							 				  sA[nt+n2-i*(i+1)/2];
			}
		}
		__syncthreads();
	}

	// Copy the factorization from shared to global memory
	for (i=n; i>0; i--){
		if (tidx<i){
			a[gb_index_x*n*n+(n-i)*(n+1)+tidx] = sA[nt+n2-i*(i+1)/2+tidx];
		}
	}
}


// Optimized for matrices bigger than 4x4 and smaller than 64x64
__global__ void LDLt_fact_hyb_k(float_gpu_st *a, int n, int ComT)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%ComT; 
    // Identifies the data concerned by the computations
	int Qt = (threadIdx.x-tidx)/ComT; 
    // The global memory access index
    int gb_index_x = Qt + blockIdx.x*(blockDim.x/ComT);
	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, j, k, n2, nt;

    n2 = (n*n+n)/2;
	nt = Qt*(n2 + n);
	
	// Copy the upper triangular part from global to shared memory
	for (i=n; i>0; i--){
		for (j=0; j<i; j+=ComT){
			if (tidx+j<i){
				sA[nt+n2-i*(i+1)/2+tidx+j] = a[gb_index_x*n*n+(n-i)*(n+1)+tidx+j];
			}
		}
	}

	__syncthreads();

	// Perform the LDLt factorization
	for(i=n; i>0; i--){
		if(tidx==0){
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2] = sA[nt+n2-i*(i+1)/2] - sA[nt+n2-k*(k+1)/2]*
									   sA[nt+n2-k*(k+1)/2+k-i]*
									   sA[nt+n2-k*(k+1)/2+k-i];
			}
		}
		__syncthreads();
		for (j=0; j<i-1; j+=ComT){
			if(tidx+j<i-1){
				sA[nt+n2-i*(i+1)/2+tidx+j+1] = sA[nt+n2-i*(i+1)/2+tidx+j+1]/sA[nt+n2-i*(i+1)/2];
				for(k=n; k>i; k--){
					sA[nt+n2-i*(i+1)/2+tidx+j+1] = sA[nt+n2-i*(i+1)/2+tidx+j+1]-sA[nt+n2-k*(k+1)/2]*
												    sA[nt+n2-k*(k+1)/2+k-i]*
												    sA[nt+n2-k*(k+1)/2+tidx+j+1]/
							 					    sA[nt+n2-i*(i+1)/2];
				}
			}
		}
		__syncthreads();
	}

	// Copy the factorization from shared to global memory
	for (i=n; i>0; i--){
		for (j=0; j<i; j+=ComT){
			if (tidx+j<i){
				a[gb_index_x*n*n+(n-i)*(n+1)+tidx+j] = sA[nt+n2-i*(i+1)/2+tidx+j];
			}
		}
	}
}


/**********************************************************************

System resolution

**********************************************************************/
// Optimized for matrices smaller than 4x4: one thread per system
__global__ void LDLt_k(float_gpu_st *a, float_gpu_st *y, int n)
{
	// Thread identifier in the grid
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, j, k, n2, n2p1;

	n2 = (n*n+n)/2;
	n2p1 = n2 + n;

	// Copy the lower triangular part from global to shared memory
	for (i=0; i<n; i++){
		for (j=0; j<=i; j++){
			sA[threadIdx.x*n2p1+i*(i+1)/2+j] = a[tidx*n*n+i*n+j];
		}
	}
	// Copy the value vector from global to shared memory
	for (i=0; i<n; i++){
		sA[threadIdx.x*n2p1+n2+i] = y[tidx*n+i];
	}

	// Perform the LDLt factorization
	for(i=0; i<n; i++){
		for(j=0; j<i; j++){
			//Mat[i*Dim+j] /= Mat[j*Dim+j];
			sA[threadIdx.x*n2p1+i*(i+1)/2+j] = sA[threadIdx.x*n2p1+i*(i+1)/2+j]/sA[threadIdx.x*n2p1+j*(j+1)/2+j];
			for(k=0; k<j; k++){
				//Mat[i*Dim+j] -= Mat[k*Dim+k]*Mat[i*Dim+k]*Mat[j*Dim+k]/Mat[j*Dim+j];
				sA[threadIdx.x*n2p1+i*(i+1)/2+j] = sA[threadIdx.x*n2p1+i*(i+1)/2+j] - sA[threadIdx.x*n2p1+k*(k+1)/2+k]*
													sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
													sA[threadIdx.x*n2p1+j*(j+1)/2+k]/
													sA[threadIdx.x*n2p1+j*(j+1)/2+j];
			}
		}
		for(k=0; k<i; k++){
			//Mat[i*Dim+i] -= Mat[k*Dim+k]*Mat[i*Dim+k]*Mat[i*Dim+k];
			sA[threadIdx.x*n2p1+i*(i+1)/2+i] = sA[threadIdx.x*n2p1+i*(i+1)/2+i] - sA[threadIdx.x*n2p1+k*(k+1)/2+k]*
												sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
												sA[threadIdx.x*n2p1+i*(i+1)/2+k]; 
		}
	}

	// Resolve the system using LDLt factorization
	for(i=0; i<n; i++){
		for(k=0; k<i; k++){
			//X[i] -= Mat[i*Dim+k]*X[k];
			sA[threadIdx.x*n2p1+n2+i] = sA[threadIdx.x*n2p1+n2+i] - sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
										 sA[threadIdx.x*n2p1+n2+k];
		}
	}
	for(i=n-1; i>=0; i--){
		//X[i] /= Mat[i*Dim+i];
		sA[threadIdx.x*n2p1+n2+i] = sA[threadIdx.x*n2p1+n2+i]/sA[threadIdx.x*n2p1+i*(i+1)/2+i];
		for(k=i+1; k<n; k++){
			//X[i] -= Mat[i*Dim+k]*X[k];
			sA[threadIdx.x*n2p1+n2+i] = sA[threadIdx.x*n2p1+n2+i] - sA[threadIdx.x*n2p1+k*(k+1)/2+i]*
										 sA[threadIdx.x*n2p1+n2+k];
		}
	}

	// Copy the solution vector from shared to global memory
	for (i=0; i<n; i++){
		y[tidx*n+i] = sA[threadIdx.x*n2p1+n2+i];
	}
}


// Optimized for matrices bigger than 64x64: x64 number of threads
__global__ void LDLt_max_k(float_gpu_st *a, float_gpu_st *y, int n)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n; 
    // Identifies the data concerned by the computations
	int Qt = (threadIdx.x-tidx)/n; 
    // The global memory access index
    int gb_index_x = Qt + blockIdx.x*(blockDim.x/n);
	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, k, n2, nt;

    n2 = (n*n+n)/2;
	nt = Qt*(n2 + n);
	
	// Copy the upper triangular part from global to shared memory
	for (i=n; i>0; i--){
		if (tidx<i){
			sA[nt+n2-i*(i+1)/2+tidx] = a[gb_index_x*n*n+(n-i)*(n+1)+tidx];
		}
	}

	// Copy the value vector from global to shared memory
	sA[nt+n2+tidx] = y[gb_index_x*n+tidx]; 
	__syncthreads();

	// Perform the LDLt factorization
	for(i=n; i>0; i--){
		if(tidx==0){
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2] = sA[nt+n2-i*(i+1)/2] - sA[nt+n2-k*(k+1)/2]*
									   sA[nt+n2-k*(k+1)/2+k-i]*
									   sA[nt+n2-k*(k+1)/2+k-i];
			}
		}
		__syncthreads();
		if(tidx<i-1){
			sA[nt+n2-i*(i+1)/2+tidx+1] = sA[nt+n2-i*(i+1)/2+tidx+1]/sA[nt+n2-i*(i+1)/2];
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2+tidx+1] = sA[nt+n2-i*(i+1)/2+tidx+1] - sA[nt+n2-k*(k+1)/2]*
											  sA[nt+n2-k*(k+1)/2+k-i]*
											  sA[nt+n2-k*(k+1)/2+tidx+1+k-i]/
							 				  sA[nt+n2-i*(i+1)/2];
			}
		}
		__syncthreads();
	}

	// Resolve the system using LDLt factorization
	for(i=0; i<n-1; i++){
		if(tidx>i){
			sA[nt+n2+tidx] = sA[nt+n2+tidx] - sA[nt+n2-(n-i)*(n-i+1)/2+tidx-i]*
							  sA[nt+n2+i];
		}
		__syncthreads();
	}
	sA[nt+n2+tidx] = sA[nt+n2+tidx]/sA[nt+n2-(n-tidx)*(n-tidx+1)/2];
	__syncthreads();
	for(i=n-1; i>0; i--){
		if(tidx<i){
			sA[nt+n2+tidx] = sA[nt+n2+tidx] - sA[nt+n2-(n-tidx)*(n-tidx+1)/2+i-tidx]*
							  sA[nt+n2+i];
		}
		__syncthreads();
	}

	// Copy the solution vector from shared to global memory
	y[gb_index_x*n+tidx] = sA[nt+n2+tidx];
}


// Optimized for matrices bigger than 4x4 and smaller than 64x64
__global__ void LDLt_hyb_k(float_gpu_st *a, float_gpu_st *y, int n, int ComT)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%ComT; 
    // Identifies the data concerned by the computations
	int Qt = (threadIdx.x-tidx)/ComT; 
    // The global memory access index
    int gb_index_x = Qt + blockIdx.x*(blockDim.x/ComT);


	cadna_init_gpu();

	// Shared memory
	extern __shared__ float_gpu_st sA[];
	// Local integers
	int i, j, k, n2, nt;

    n2 = (n*n+n)/2;
	nt = Qt*(n2 + n);
	
	// Copy the upper triangular part from global to shared memory
	for (i=n; i>0; i--){
		for (j=0; j<i; j+=ComT){
			if (tidx+j<i){
				sA[nt+n2-i*(i+1)/2+tidx+j] = a[gb_index_x*n*n+(n-i)*(n+1)+tidx+j];
			}
		}
	}

	// Copy the value vector from global to shared memory
	for (j=0; j<n; j+=ComT){
		if (tidx+j<n){
			sA[nt+n2+tidx+j] = y[gb_index_x*n+tidx+j];
		}
	}
	__syncthreads();

	// Perform the LDLt factorization
	for(i=n; i>0; i--){
		if(tidx==0){
			for(k=n; k>i; k--){
				sA[nt+n2-i*(i+1)/2] = sA[nt+n2-i*(i+1)/2] - sA[nt+n2-k*(k+1)/2]*
									   sA[nt+n2-k*(k+1)/2+k-i]*
									   sA[nt+n2-k*(k+1)/2+k-i];
			}
		}
		__syncthreads();
		for (j=0; j<i-1; j+=ComT){
			if(tidx+j<i-1){
				sA[nt+n2-i*(i+1)/2+tidx+j+1] = sA[nt+n2-i*(i+1)/2+tidx+j+1]/sA[nt+n2-i*(i+1)/2];
				for(k=n; k>i; k--){
					sA[nt+n2-i*(i+1)/2+tidx+j+1] = sA[nt+n2-i*(i+1)/2+tidx+j+1] - sA[nt+n2-k*(k+1)/2]*
												    sA[nt+n2-k*(k+1)/2+k-i]*
												    sA[nt+n2-k*(k+1)/2+tidx+j+1+k-i]/
							 					    sA[nt+n2-i*(i+1)/2];
				}
			}
		}
		__syncthreads();
	}

	// Resolve the system using LDLt factorization
	for(i=0; i<n-1; i++){
		for (j=n-ComT; j>-ComT; j-=ComT){
			if(tidx+j>i){
				sA[nt+n2+tidx+j] = sA[nt+n2+tidx+j] - sA[nt+n2-(n-i)*(n-i+1)/2+tidx+j-i]*
								    sA[nt+n2+i];
			}
		}
		__syncthreads();
	}
	for (j=0; j<n; j+=ComT){
		if (tidx+j<n){
			sA[nt+n2+tidx+j] = sA[nt+n2+tidx+j]/sA[nt+n2-(n-tidx-j)*(n-tidx-j+1)/2];
		}
	}
	__syncthreads();
	for(i=n-1; i>0; i--){
		for (j=0; j<i; j+=ComT){
			if(tidx+j<i){
				sA[nt+n2+tidx+j] = sA[nt+n2+tidx+j] - sA[nt+n2-(n-tidx-j)*(n-tidx-j+1)/2+i-tidx-j]*
									sA[nt+n2+i];
			}
		}
		__syncthreads();
	}

	// Copy the solution vector from shared to global memory
	for (j=0; j<n; j+=ComT){
		if (tidx+j<n){
			y[gb_index_x*n+tidx+j] = sA[nt+n2+tidx+j];
		}
	}
}