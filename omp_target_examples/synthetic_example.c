#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#pragma omp declare target
int omp_get_thread_num(void);
int omp_get_num_threads(void);
void omp_set_num_threads(int);
void omp_set_nested(int);
int omp_get_nested(void);
int usleep(useconds_t);
#pragma omp end declare target

int main()
{
  MPI_Init(NULL, NULL);

  int i = 0, j = 0, k = 0;
  int values[200], values2[200];

  for (i = 0; i < 200; ++i) {
    values[i] = i;
    values2[i] = -i;
  }

  
  printf("host: nesting = %d num threads = %d max thread = %d\n",
	omp_get_nested(), omp_get_num_threads(), omp_get_thread_limit());
  
#pragma omp target map(tofrom:values[0:200]) map(to:values2[0:200])
{
    printf("nesting = %d max thread = %d\n", omp_get_nested(), omp_get_thread_limit());
    omp_set_nested(1);

#pragma omp parallel for
    for (i = 0; i < 2; ++i) {
      int tNum = omp_get_thread_num();
#pragma omp parallel for
      for (j = 0; j < 10 + (1-tNum)*10; ++j) {
        values[j] += values2[j];
	usleep(100);
      }
    }
#pragma omp parallel for
    for (i = 0; i < 2; ++i) {
      int tNum = omp_get_thread_num();
#pragma omp parallel for
      for (j = 0; j < 10 + tNum*10; ++j) {
        values[j] += values2[j];
        usleep(100);
      }
    }
}
  
  MPI_Finalize();
  return 0;
}
