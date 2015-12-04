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

/**
 * @brief Does one Jacobi iteration on A_d writing the results to
 *        Anew_d on all interior points of the domain.
 *
 * The Jacobi iteration solves the poission equation with diriclet
 * boundary conditions and a zero right hand side and returns the max
 * norm of the residue, executes synchronously.
 *
 * @param[in] A_d            pointer to device memory holding the 
 *                           solution of the last iteration including
 *                           boundary.
 * @param[out] Anew_d        pointer to device memory were the updates
 *                           solution should be written
 * @param[in] n              number of points in y direction
 * @param[in] m              number of points in x direction
 * @param[in,out] residue_d  pointer to a single float value in device
 * 			     memory, needed a a temporary storage to
 * 			     calculate the max norm of the residue.
 * @return		     the residue of the last iteration
 */
float launch_jacobi_kernel( const float* const A_d, float* const Anew_d, 
                            const int n, const int m, float* const residue_d );


/**
 * @brief Copies all inner points from Anew_d to A_d, executes
 *        asynchronously.
 *
 * @param[out] A_d    pointer to device memory holding the solution of
 * 		      the last iteration including boundary which
 * 		      should be updated with Anew_d
 * @param[in] Anew_d  pointer to device memory were the updated
 *                    solution is saved
 * @param[in] n       number of points in y direction
 * @param[in] m       number of points in x direction
 */
void launch_copy_kernel( float* const A_d, const float* const Anew_d, 
                         const int n, const int m );


int  handle_command_line_arguments(int argc, char** argv);
int  init_mpi(int argc, char** argv);
void init_host();
void init_cuda();

void finalize_mpi();
void finalize_host();
void finalize_cuda();

void start_timer();
void stop_timer();

void jacobi();


int n, m;
int rank=0;
int size=1;

int iter = 0;
int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

int n_global;

float* A;
float* Anew;
float* y0;

float* A_d;
float* Anew_d;
float* residue_d;

#ifdef USE_MPI
float* sendBuffer;
float* recvBuffer;
#endif //USE_MPI

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
  char *str = NULL;
  if ((str = getenv("PMI_RANK")) != NULL)
    {
      rank = atoi(str);
    }

  cudaSetDevice( rank );

#ifdef USE_MPI
  if ( init_mpi(argc, argv) )
    {
      return 1;
    }
#endif //USE_MPI

  if ( handle_command_line_arguments(argc, argv) )
    {
      return -1;
    }

  init_host();
  init_cuda();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif //USE_MPI
    
  start_timer();

  // Main calculation
  jacobi();

  stop_timer();

  finalize_cuda();
  finalize_host();

#ifdef USE_MPI
  finalize_mpi();
#endif //USE_MPI
}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{
  while ( residue > tol && iter < iter_max )
    {
      residue = launch_jacobi_kernel(  A_d, Anew_d, n, m, residue_d );

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
              MPI_Sendrecv( Anew_d+(n-2)*m+1, m-2, MPI_FLOAT, 1, 0, A_d+(n-1)*m+1 , m-2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status );
            }
          else
            {
              MPI_Sendrecv( Anew_d + 1*m+1, m-2, MPI_FLOAT, 0, 0, A_d+0*m+1, m-2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );
            }
			
        }
#endif //USE_MPI
		
      launch_copy_kernel(A_d,Anew_d,n,m);

      if(rank == 0 && iter % 100 == 0)
        printf("%5d, %0.6f\n", iter, residue);

      iter++;
    }

  cudaMemcpy( A, A_d, m*n*sizeof(float), cudaMemcpyDeviceToHost );
}


/********************************/
/**** Initialization routines ***/
/********************************/

#ifdef USE_MPI
int init_mpi(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if ( size != 1 && size != 2 )
    {
      if ( rank == 0)
        printf("Error: %s can only run with 1 or 2 processes!\n",argv[0]);
      return 1;
    }

  return 0;
}
#endif //USE_MPI

void init_host()
{
  A	= (float*) malloc( n*m * sizeof(float) );
  Anew	= (float*) malloc( n*m * sizeof(float) );
  y0	= (float*) malloc( n   * sizeof(float) );

#ifdef OMP_MEMLOCALTIY
#pragma omp parallel for shared(A,Anew,m,n)
  for( int j = 0; j < n; j++)
    {
      for( int i = 0; i < m; i++ )
        {
          Anew[j *m+ i] 	= 0.0f;
          A[j *m+ i] 		= 0.0f;
        }
    }
#else
  memset(A, 0, n * m * sizeof(float));
  memset(Anew, 0, n * m * sizeof(float));
#endif //OMP_MEMLOCALTIY

  // set boundary conditions
#pragma omp parallel for shared(A,m,n,rank,size)
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
}

void init_cuda()
{
  cudaMalloc( (void**)&A_d, n*m * sizeof(float) );
  cudaMalloc( (void**)&Anew_d, n*m * sizeof(float) );
  cudaMalloc( (void**)&residue_d, sizeof(float) );

  cudaMemcpy( A_d, A, m*n*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( Anew_d, Anew, m*n*sizeof(float), cudaMemcpyHostToDevice );
}

int handle_command_line_arguments(int argc, char** argv)
{
  if ( argc > 3 )
    {
      if ( rank == 0)
        printf( "usage: %s [n] [m]\n", argv[0] );
      return 1;
    }

  n = 4096;
  if ( argc >= 2 )
    {
      n = atoi( argv[1] );
      if ( n <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of rows (n=%i) needs to positive!\n",n);
          return 1;
        }
    }
  if ( size == 2 && n%2 != 0 )
    {
      if ( rank == 0)
        printf("Error: The number of rows (n=%i) needs to be devisible by 2 if two processes are used!\n",n);
      return 1;
    }
  m = n;
  if ( argc >= 3 )
    {
      m = atoi( argv[2] );
      if ( m <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of columns (m=%i) needs to positive!\n",m);
          return 1;
        }
    }

  n_global = n;

  if ( size == 2 )
    {
      //Do a domain decomposition and add one row for halo cells
      n = n/2 + 1;
    }

  if ( rank == 0 )
    {
      struct cudaDeviceProp devProp;
      cudaGetDeviceProperties( &devProp, rank );
      printf("Jacobi relaxation Calculation: %d x %d mesh with "
             "%d processes and one %s for each process.\n"
             , n_global, m, size, devProp.name);
    }

  return 0;
}


/********************************/
/****  Finalization routines  ***/
/********************************/

#ifdef USE_MPI
void finalize_mpi()
{
  MPI_Finalize();
}
#endif //USE_MPI

void finalize_host()
{
  free(y0);
  free(Anew);
  free(A);
}

void finalize_cuda()
{
  cudaDeviceSynchronize();

  cudaFree( residue_d );
  cudaFree(Anew_d);
  cudaFree(A_d);
}

/********************************/
/****    Timing functions     ***/
/********************************/
void start_timer()
{
#ifdef USE_MPI
  starttime = MPI_Wtime();
#else
  starttime = omp_get_wtime();
#endif //USE_MPI
}

void stop_timer()
{
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = MPI_Wtime() - starttime;
#else
  runtime = omp_get_wtime() - starttime;
#endif //USE_MPI

  if (rank == 0)
    printf(" total: %f s\n", runtime);
}

