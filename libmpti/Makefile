CC = icc
MPICC = mpiicc
SCOREP = scorep --mpp=mpi --thread=omp:ancestry #--keep-files

# Folder on MICs where the library can be put (MUST be accessible by user "micuser", so make sure permissions are set)
MIC_0_DIR = /tmp
MIC_1_DIR = /tmp
# Host names of the MICs
MIC_0_HOST = mic0
MIC_1_HOST = mic1

# Folder of OMPT library for host and MIC
OMPT_LIB_HOST = ${INTELOMPRT_ROOT}/lib
OMPT_LIB_MIC = ${INTELOMPRT_ROOT}/lib_mic

#OMPT Include dir (path to ompt.h)
OMPT_INCDIR = $(OMPT_LIB_HOST)

#Runtime Settings
SCOREP_SETTINGS = SCOREP_EXPERIMENT_DIRECTORY=trace SCOREP_TOTAL_MEMORY=500M
OMP_SETTINGS = OMP_NESTED=1 OMP_NUM_THREADS=2 MIC_OMP_NUM_THREADS=4
PRELOAD_SETTINGS = MIC_ENV_PREFIX=MIC MIC_0_LD_PRELOAD=$(MIC_0_DIR)/libmpti.so MIC_1_LD_PRELOAD=$(MIC_1_DIR)/libmpti.so
############################################################################

C_FLAGS = -openmp -I$(OMPT_INCDIR) -g
LD_FLAGS = $(OMPT_LIB_HOST)/libiomp5.so 
MIC_FLAGS = -mmic -I$(OMPT_INCDIR) -shared -fPIC -L$(OMPT_LIB_MIC) -openmp

BINARIES = omp_host libmpti.so

SRC = omp_target_mic.c
SRC = synthetic_example.c

all: $(BINARIES)

omp_host: $(SRC) libmpti.so
	$(SCOREP) $(MPICC) $(C_FLAGS) -o $@.o -c $<
	$(SCOREP) $(MPICC) $(LD_FLAGS) -o $@ $@.o
	rm -f $@*.o
	
libmpti.so: mpti.cpp
	$(CC) $(MIC_FLAGS) -o $@ $+

run: all
	scp libmpti.so $(MIC_0_HOST):$(MIC_0_DIR)
	scp libmpti.so $(MIC_1_HOST):$(MIC_1_DIR)
	ssh $(MIC_0_HOST) chmod o+rx $(MIC_0_DIR)/libmpti.so
	ssh $(MIC_1_HOST) chmod o+rx $(MIC_1_DIR)/libmpti.so
	$(SCOREP_SETTINGS) $(OMP_SETTINGS) $(PRELOAD_SETTINGS) MIC_LD_LIBRARY_PATH=$(OMPT_LIB_MIC):${MIC_LD_LIBRARY_PATH} mpirun -n 1 ./omp_host
	ssh $(MIC_0_HOST) rm $(MIC_0_DIR)/libmpti.so
	ssh $(MIC_1_HOST) rm $(MIC_1_DIR)/libmpti.so
	
clean:
	rm -f $(BINARIES) *.mod.c *.opari.inc *.opari.c *pomp_init.c *.input.c *.prep.c *.o
	rm -rf scorep-*
