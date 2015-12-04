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

/*
* Robert Dietrich, Felix Schmitt, Alexander Grund, Jonas Stolle:
* We created this file from jacobi_openmp.c and introduced OpenMP target 
* directives to execute the jacobi and the copy kernels on the device.
*/

#ifdef USE_MPI
#include <mpi.h>
#endif //USE_MPI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

int  handle_command_line_arguments(int argc, char** argv);
int  init_mpi(int argc, char** argv);
void init_host(float* A, float* Anew);

void finalize_mpi();
void finalize_host(float** A, float** Anew);

void start_timer();
void stop_timer();

void jacobi(float* A, float* Anew);


int n;
int n_global;

int   n_cpu;
float lb;

int   cpu_start, cpu_end;
#pragma omp declare target
int m;
int   gpu_start, gpu_end;
int rank=0;
#pragma omp end declare target
int size=1;

int iter = 0;
int iter_max = 100;//1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* y0;

#ifdef USE_MPI
float* sendBuffer;
float* recvBuffer;
#endif //USE_MPI

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
  float* A;
  float* Anew;

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

  A = (float*) malloc( n*m * sizeof(float) );
  Anew  = (float*) malloc( n*m * sizeof(float) );
  init_host(A, Anew);
  #pragma omp target data map(to:A[gpu_start-1:m*(n-n_cpu)], Anew[gpu_start-1:m*(n-n_cpu)])
  {
  #ifdef USE_MPI
    /* This has do be done after handling command line arguments */
    sendBuffer = (float*) malloc ( (m-2) * sizeof(float) );
    recvBuffer = (float*) malloc ( (m-2) * sizeof(float) );

    MPI_Barrier(MPI_COMM_WORLD);
  #endif //USE_MPI

    start_timer();

    // Main calculation
    jacobi(A, Anew);

    stop_timer();
  }
  
  finalize_host(&A, &Anew);
  
#ifdef USE_MPI
  finalize_mpi();
#endif //USE_MPI
}   

#pragma omp declare target
void jacobi_kernel(const float* restrict const A, float* restrict const Anew, float* restrict const residue, int jStart, int jEnd){
  float residue_tmp = 0.0f;

double start = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp for reduction(max:residue_tmp)
    for( int j = jStart; j < jEnd; j++){
        for( int i = 1; i < m-1; i++ ){
            Anew[j *m+ i] = 0.25f * ( A[j     *m+ (i+1)] + A[j     *m+ (i-1)]
                                      +    A[(j-1) *m+ i]     + A[(j+1) *m+ i]);
            float tmp = fabsf(Anew[j *m+ i]-A[j *m + i]);
            if(tmp > residue_tmp) residue_tmp = tmp;
        }
    }
  }
  if(omp_get_max_threads() != 24) 
    printf("time=%gms\n",(omp_get_wtime()-start)*1000.);
  *residue = residue_tmp;
}

void copy_kernel(float* restrict const A, const float* restrict const Anew, int jStart, int jEnd){
  #pragma omp parallel for
  for( int j = jStart; j < jEnd; j++)
  {
    #pragma omp simd
    for( int i = 1; i < m-1; i++ )
    {
      A[j *m+ i] = Anew[j *m+ i];
    }
  }
}
#pragma omp end declare target

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi(float* A, float* Anew){
  omp_set_nested(1);

  while ( residue > tol && iter < iter_max ){
      float residueHost=0.0f, residueTarget=0.0f;
      #pragma omp parallel num_threads(2)
      {
        #pragma omp master
        {
          #pragma omp task
          {
	          //double start=omp_get_wtime();
            jacobi_kernel(A, Anew, &residueHost, cpu_start, cpu_end);
            //printf("host=%gms\n",(omp_get_wtime()-start)*1000.f);
          }

          #pragma omp task
          {
	          //double start=omp_get_wtime();
            #pragma omp target map(from:residueTarget)
              jacobi_kernel(A, Anew, &residueTarget, gpu_start, gpu_end);
            //printf("target=%gms\n",(omp_get_wtime()-start)*1000.f);
          }
        }
      }
      residue = fmaxf(residueHost, residueTarget);


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
          if ( rank == 0 ){
            #pragma omp target update to(Anew[cpu_start*m+1:m-2])
            #pragma omp target update from(Anew[(cpu_start-1)*m+1:m-2])
          }else{
            #pragma omp target update to(Anew[cpu_end*m+1:m-2])
            #pragma omp target update from(Anew[(cpu_end+1)*m+1:m-2])
          }
        }

      #pragma omp parallel num_threads(2)
      {
        #pragma omp master
        {
          #pragma omp task
          copy_kernel(A, Anew, cpu_start+((rank==0)?-1:0), cpu_end+((rank==0)?0:1));

          #pragma omp task
          {
            #pragma omp target
            copy_kernel(A, Anew, gpu_start+((rank==0)?0:-1), gpu_end+((rank==0)?1:0));
          }
        }
      }

      if(rank == 0 && iter % 100 == 0)
        printf("%5d, %0.6f\n", iter, residue);

      iter++;
    }

  #pragma omp target update from(A[gpu_start:m*(n-n_cpu-1)-2])
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

void init_host(float* A, float* Anew)
{
  iter = 0;
  residue = 1.0f;

  // Index of first gpu element in HOST array
  gpu_start = rank==0 ? 1      : n_cpu+1;
  gpu_end   = gpu_end = gpu_start + n-n_cpu-2;
  cpu_start = rank==0 ? n-n_cpu +1 : 1;
  cpu_end   = cpu_start + n_cpu - 2;
#pragma omp target update to(gpu_start, gpu_end, rank,m)

  y0	= (float*) malloc( n   * sizeof(float) );

#ifdef OMP_MEMLOCALTIY
#pragma omp parallel for shared(A,Anew,m,n)
  for( int j = cpu_start; j < cpu_end; j++)
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
#pragma omp parallel for
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

#pragma omp parallel for
  for (int i = 1; i < m; i++)
    {
      if (rank == 0)
        Anew[0     *m+ i] = 0.f;
      if (rank == 1 || size == 1)
        Anew[(n-1) *m+ i] = 0.f;
    }
#pragma omp parallel for
  for (int j = 1; j < n; j++)
    {
      Anew[j *m+ 0] = y0[j];
      Anew[j *m+ (m-1)] = y0[j]*expf(-pi);
    }
}

int handle_command_line_arguments(int argc, char** argv)
{
  if ( argc > 4 )
    {
      if ( rank == 0)
        printf( "usage: %s [n] [m] [lb]\n", argv[0] );
      return -1;
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
        printf("Error: The number of rows (n=%i) needs to be divisible by 2 if two processes are used!\n",n);
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
  
  n_global = n;

  if ( size == 2 )
    {
      //Do a domain decomposition and add one row for halo cells
      n = n/2 + 1;
    }

  n_cpu = lb*n;

  if ( rank == 0 )
    {
      #pragma omp parallel
      {
        #pragma omp master
        {
          int targetThreads;
          #pragma omp target map(from:targetThreads)
          {
            #pragma omp parallel
            #pragma omp master
              targetThreads = omp_get_num_threads();
          }

          printf("Jacobi relaxation Calculation: %d x %d mesh "
                 "with %d processes and %d threads + one Target with %d threads for "
                 "each process.\n",
                 n_global, m,size,omp_get_num_threads(),targetThreads);
          printf("\t%d of %d local rows are calculated on the "
                 "CPU to balance the load between the CPU and "
                 "the Target.\n",
                 n_cpu, n);
        }
      }
    }
  return 0;
}


/********************************/
/****  Finalization routines  ***/
/********************************/

#ifdef USE_MPI
void finalize_mpi()
{
  free( recvBuffer );
  free( sendBuffer );

  MPI_Finalize();
}
#endif //USE_MPI

void finalize_host(float** A, float** Anew)
{
  free(y0);
  free(*Anew);
  free(*A);
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

