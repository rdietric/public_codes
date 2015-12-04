PREP=scorep --thread=omp:pomp_tpd --mpp=mpi
#PREP=

MPICC=$(PREP) mpicc
CC=$(PREP) gcc
NVCC=$(PREP) nvcc

OMPFLAG=-fopenmp

CFLAGS=-std=c99 -DOMP_MEMLOCALTIY
CUDA_PATH=${CUDA_ROOT}
CUDACFLAGS=-I$(CUDA_PATH)/include
MPICFLAGS=-DUSE_MPI
NVCCFLAGS=-O3 -arch=sm_20 -Xcompiler -march=native

LDFLAGS=-lm
# -lstdc++ -lmpi
CUDALDFLAGS=-L$(CUDA_PATH)/lib64 -lcudart
#TARGET_LDFLAGS=-lm /home/alex/libs/libiomp5.so

BINDIR=.
JACOBI_OPENMP=$(BINDIR)/jacobi_openmp
JACOBI_CUDA=$(BINDIR)/jacobi_cuda
JACOBI_MPIOPENMP=$(BINDIR)/jacobi_mpi+openmp
JACOBI_MPICUDA=$(BINDIR)/jacobi_mpi+openmp+cuda
JACOBI_MPIOMPTARGET=$(BINDIR)/jacobi_mpi+openmp+target
BINARIES=$(JACOBI_OPENMP) $(JACOBI_MPIOPENMP) $(JACOBI_MPIOMPTARGET)
#$(JACOBI_CUDA) $(JACOBI_MPICUDA)

# Folder on MIC(s) that accesible from host and MIC side (will be created if not existing)
MIC_0_DIR = /mic0fs/${USER}/libs
MIC_1_DIR = /mic1fs/${USER}/libs

#Runtime Settings
SCOREP_SETTINGS = SCOREP_EXPERIMENT_DIRECTORY=scorep-trace SCOREP_TOTAL_MEMORY=500M SCOREP_OMPT_ENABLE=1
# SCOREP_DEBUG=openmp
OMP_SETTINGS = OMP_NESTED=1 OMP_NUM_THREADS=2 MIC_OMP_NUM_THREADS=4 MIC_OMP_NESTED=1
PRELOAD_SETTINGS = MIC_ENV_PREFIX=MIC MIC_0_LD_PRELOAD=$(MIC_0_DIR)/libmpti.so MIC_1_LD_PRELOAD=$(MIC_1_DIR)/libmpti.so

all: $(BINARIES)

jacobi_cuda_kernel.o: jacobi_cuda_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

jacobi_openmp.o: jacobi_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAG) -c $< -o $@
	
jacobi_cuda.o: jacobi_cuda.c
	$(CC) $(CFLAGS) $(OMPFLAG) $(CUDACFLAGS) -c $< -o $@
	
jacobi_mpi+openmp.o: jacobi_openmp.c
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(MPICFLAGS) -c $< -o $@
	
jacobi_mpi+cuda.o: jacobi_cuda.c
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(MPICFLAGS) $(CUDACFLAGS) -c $< -o $@
	
jacobi_mpi+target.o: jacobi_target.c
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(MPICFLAGS) -c $< -o $@
	
$(JACOBI_OPENMP): jacobi_openmp.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(OMPFLAG) $^ -o $@
	
$(JACOBI_CUDA): jacobi_cuda.o jacobi_cuda_kernel.o
	$(CC) $(LDFLAGS) $(OMPFLAG) $(CUDALDFLAGS) $^ -o $@

$(JACOBI_MPIOPENMP): jacobi_mpi+openmp.o
	$(MPICC) $(TARGET_LDFLAGS) $^ -o $@
	
$(JACOBI_MPICUDA): jacobi_mpi+cuda.o jacobi_cuda_kernel.o
	$(MPICC) $(LDFLAGS) $(OMPFLAG) $(CUDALDFLAGS) $^ -o $@
	
$(JACOBI_MPIOMPTARGET): jacobi_mpi+target.o
	$(MPICC) $(TARGET_LDFLAGS) $^ -o $@

run: $(JACOBI_MPIOMPTARGET)
	$(SCOREP_SETTINGS) $(OMP_SETTINGS) $(PRELOAD_SETTINGS) MIC_LD_LIBRARY_PATH=$(OMPT_LIB_MIC):${MIC_LD_LIBRARY_PATH} mpirun -np 1 ./jacobi_mpi+openmp+target
	
clean:
	rm -rf *.o *.opari.* *.pomp_init.* *.prep.* *.init.* $(BINARIES)
	rm -rf scorep-trace*
