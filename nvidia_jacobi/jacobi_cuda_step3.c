/*
* Copyright 2012 NVIDIA Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#endif //USE_MPI
#include <omp.h>
#include <cuda_runtime.h>

#ifdef VTRACE
    #include "vt_user.h"
#else
    #define VT_USER_START(a)
    #define VT_USER_END(a)
#endif //VTRACE

float launch_jacobi_kernel( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d );
void launch_jacobi_kernel_async( const float* const A_d, float* const Anew_d, const int n, const int m, float* const residue_d );
float wait_jacobi_kernel( float* const residue_d );
void launch_copy_kernel( float* const A_d, const float* const Anew_d, const int n, const int m );

int main(int argc, char** argv)
{
	int rank=0;
	int size=1;

#ifdef USE_MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif //USE_MPI
	
	if ( argc > 4 )
	{
		if ( rank == 0)
			printf( "usage: %s [n] [m] [lb]\n", argv[0] );
		return -1;
	}

	cudaSetDevice( rank );

	if ( size != 1 && size != 2 )
	{
		if ( rank == 0)
			printf("Error: %s can only run with 1 or 2 processes!\n",argv[0]);
		return -1;
	}

    int n = 4096;
    if ( argc >= 2 )
    {
    	n = atoi( argv[1] );
    	if ( n <= 0 )
		{
			if ( rank == 0 )
				printf("Error: The number of rows (n=%i) needs to positive!\n",n);
			return -1;
		}
    }
    if ( size == 2 && n%2 != 0 )
    {
    	if ( rank == 0)
			printf("Error: The number of rows (n=%i) needs to be devisible by 2 if two processes are used!\n",n);
		return -1;
    }
    int m = n;
    if ( argc >= 3 )
	{
		m = atoi( argv[2] );
		if ( m <= 0 )
		{
			if ( rank == 0 )
				printf("Error: The number of columns (m=%i) needs to positive!\n",m);
			return -1;
		}

	}
    int iter_max = 1000;
    float lb = 0.0f;
    if ( argc == 4 )
    {
    	lb = atof( argv[3] );
    	if ( lb < 0.0f || lb > 1.0f )
    	{
    		if ( rank == 0 )
				printf("Error: The load balancing factor (lb=%0.2f) needs to be in [0:1]!\n",lb);
			return -1;
    	}
    }
    int n_cpu = lb*n;

    const float pi = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float residue = 1.0f;

    int global_n = n;
    if ( size == 2 )
    {
    	//Do a domain decomposition and add one row for halo cells
    	n = n/2 + 1;
    }

    float* A	= (float*) malloc( n*m * sizeof(float) );
    float* Anew	= (float*) malloc( n*m * sizeof(float) );
    float* y0	= (float*) malloc( n   * sizeof(float) );

    float* A_d;
	cudaMalloc( (void**)&A_d, n*m * sizeof(float) );
	float* Anew_d;
	cudaMalloc( (void**)&Anew_d, n*m * sizeof(float) );

    float* residue_d;
    cudaMalloc( (void**)&residue_d, sizeof(float) );

#ifdef USE_MPI
    float* sendBuffer = (float*) malloc ( (m-2) * sizeof(float) );
    float* recvBuffer = (float*) malloc ( (m-2) * sizeof(float) );
#endif //USE_MPI

    int j_start = n_cpu == n ? 1 : n-n_cpu;
	if ( rank == 1 )
		j_start = 1;

	int j_end = n-1;
	if ( rank == 1 )
		j_end = n_cpu == n ? n-1 : n_cpu+1;

#ifdef OMP_MEMLOCALTIY
	#pragma omp parallel for  shared(A,Anew,m,n,j_start,j_end)
	for( int j = j_start; j < j_end; j++)
	{
		for( int i = 0; i < m; i++ )
		{
			Anew[j *m+ i] 	= 0.0f;
			A[j *m+ i] 		= 0.0f;
		}
	}
#endif //OMP_MEMLOCALTIY

	memset(A, 0, n * m * sizeof(float));
	memset(Anew, 0, n * m * sizeof(float));

    // set boundary conditions
	#pragma omp parallel for  shared(A,m,n,rank,size)
    for (int i = 0; i < m; i++)
    {
    	//Set top boundary condition only for rank 0 (rank responsible of the upper halve of the domain)
    	if ( rank == 0 )
    		A[0	    *m+ i] = 0.f;
    	//Set bottom boundary condition only for rank 1 (rank responsible of the lower halve of the domain)
    	if ( rank == 0 || size == 1 )
    		A[(n-1) *m+ i] = 0.f;
    }

    int j_offset = 0;
    if ( size == 2 && rank == 1 )
	{
		j_offset = n-2;
	}
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * (j_offset + j) / (n-1));
        A[j *m+ 0] = y0[j];
        A[j *m+ (m-1)] = y0[j]*expf(-pi);
    }

    if ( rank == 0 )
    {
    	struct cudaDeviceProp devProp;
    	cudaGetDeviceProperties( &devProp, rank );

		#pragma omp parallel
		{
			#pragma omp master
			{
				if ( n_cpu > 0 )
				{
					printf("Jacobi relaxation Calculation: %d x %d mesh with %d processes and %d threads + one %s for each process.\n", global_n, m,size,omp_get_num_threads(),devProp.name);
					printf("\t%d of %d local rows are calculated on the CPU to balance the load between the CPU and the GPU.\n", n_cpu, n);
				}
				else
				{
					printf("Jacobi relaxation Calculation: %d x %d mesh with %d processes and one %s for each process.\n", global_n, m,size,devProp.name);
				}
			}
		}
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    double starttime = MPI_Wtime();
#else
    double starttime = omp_get_wtime();
#endif //USE_MPI
    int iter = 0;


	#pragma omp parallel for  shared(Anew,m,n,rank,size)
    for (int i = 1; i < m; i++)
    {
    	if (rank == 0)
    		Anew[0     *m+ i] = 0.f;
    	if (rank == 1 || size == 1)
    		Anew[(n-1) *m+ i] = 0.f;
    }
	#pragma omp parallel for  shared(Anew,y0,m,n)
    for (int j = 1; j < n; j++)
    {
		Anew[j *m+ 0] = y0[j];
		Anew[j *m+ (m-1)] = y0[j]*expf(-pi);
    }

	if ( rank == 0 )
	{
		cudaMemcpy( A_d, A, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( Anew_d, Anew, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );
	}
	else
	{
		cudaMemcpy( A_d, A+(j_end-1)*m, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( Anew_d, Anew+(j_end-1)*m, m*(n-n_cpu)*sizeof(float), cudaMemcpyHostToDevice );
	}

    while ( residue > tol && iter < iter_max )
    {
    	VT_USER_START("jacobi_iteration");
    	residue = 0.0f;
    	launch_jacobi_kernel_async( A_d, Anew_d, n-n_cpu, m, residue_d );

    	VT_USER_START("jacobi_omp");
		#pragma omp parallel  shared(m, n, Anew, A, residue, j_start,j_end)
        {
        	float my_residue = 0.f;

			#pragma omp for nowait
			for( int j = j_start; j < j_end; j++)
			{
				for( int i = 1; i < m-1; i++ )
				{
					//Jacobi is Anew[j*m+i] = 1.0/1.0*(rhs[j*m+i] -
					//                        (                           -0.25f*A[(j-1) *m+ i]
					//                          -0.25f*A[j     *m+ (i+1)]                        -0.25f*A[j     *m+ (i-1)]
					//                                                    -0.25f*A[(j+1) *m+ i]));
					//rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
					// =>
					Anew[j *m+ i] = 0.25f * ( A[j     *m+ (i+1)] + A[j     *m+ (i-1)]
										 +    A[(j-1) *m+ i]     + A[(j+1) *m+ i]);
					//Calculate residue of A
					//residue =
					//   rhs[j*m+i] -  (                           -0.25f*A[(j-1) *m+ i]
					//                   -0.25f*A[j     *m+ (i+1)] +1.00f*A[j     *m+ i]  -0.25f*A[j     *m+ (i-1)]
					//                                             -0.25f*A[(j+1) *m+ i]));
					//rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
					// =>
					//residue =  Anew[j *m+ i]-A[j *m + i]
					my_residue = fmaxf( my_residue, fabsf(Anew[j *m+ i]-A[j *m + i]));
				}
			}

			#pragma omp critical
			{
				residue = fmaxf( my_residue, residue);
			}
        }
        VT_USER_END("jacobi_omp");

        residue = fmaxf( residue, wait_jacobi_kernel( residue_d ) );

#ifdef USE_MPI
        float globalresidue = 0.0f;
		MPI_Allreduce( &residue, &globalresidue, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD );
		residue = globalresidue;
#endif //USE_MPI

#ifdef USE_MPI
		if ( size == 2 )
		{
			MPI_Status status;
			if ( rank == 0)
			{
				MPI_Sendrecv( Anew+(n-2)*m+1, m-2, MPI_FLOAT, 1, 0, A+(n-1)*m+1 , m-2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status );
			}
			else
			{
				MPI_Sendrecv( Anew + 1*m+1, m-2, MPI_FLOAT, 0, 0, A+0*m+1, m-2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );
			}
		}
#endif //USE_MPI
		
		if ( n_cpu > 0 && n_cpu < n )
		{
			VT_USER_START("update_host_device");
			if ( rank == 0 )
			{
				cudaMemcpy( A_d+j_start*m+1, Anew+j_start*m+1, (m-2)*sizeof(float), cudaMemcpyHostToDevice );
				cudaMemcpy( A+(j_start-1)*m+1, Anew_d+(j_start-1)*m+1, (m-2)*sizeof(float), cudaMemcpyDeviceToHost );
			}
			else
			{
				cudaMemcpy( A_d+0*m+1, Anew+(j_end-1)*m+1, (m-2)*sizeof(float), cudaMemcpyHostToDevice );
				cudaMemcpy( A+(j_end)*m+1, Anew_d+0*m+1, (m-2)*sizeof(float), cudaMemcpyDeviceToHost );
			}
			VT_USER_END("update_host_device");
		}

		launch_copy_kernel(A_d,Anew_d,n-n_cpu,m);

		VT_USER_START("copy_omp");
		#pragma omp parallel for  shared(m, n, Anew, A, j_start,j_end)
		for( int j = j_start; j < j_end; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j *m+ i] = Anew[j *m+ i];
            }
        }
		VT_USER_END("copy_omp");

        if(rank == 0 && iter % 100 == 0)
        	printf("%5d, %0.6f\n", iter, residue);

        iter++;
        VT_USER_END("jacobi_iteration");
    }
	if ( rank == 0 )
	{
		cudaMemcpy( A+1*m+1, A_d+1*m+1, (m*(n-n_cpu-1)-2)*sizeof(float), cudaMemcpyDeviceToHost );
	}
	else
	{
		cudaMemcpy( A+j_end*m+1, A_d+1*m+1, (m*(n-n_cpu-1)-2)*sizeof(float), cudaMemcpyDeviceToHost );
	}

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    double runtime = MPI_Wtime() - starttime;
#else
    double runtime = omp_get_wtime() - starttime;
#endif //USE_MPI

    if (rank == 0)
    	printf(" total: %f s\n", runtime);
    
    cudaDeviceSynchronize();

#ifdef USE_MPI
    free( recvBuffer );
    free( sendBuffer );
#endif //USE_MPI

    cudaFree( residue_d );
    cudaFree(Anew_d);
    cudaFree(A_d);

    free(y0);
	free(Anew);
	free(A);

#ifdef USE_MPI
    MPI_Finalize();
#endif //USE_MPI
}

