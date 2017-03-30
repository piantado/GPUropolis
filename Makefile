## NOTE: This CUDA version must be set to your local version or it will not compile!
CUDA_VERSION=8.0

CUDA_SAMPLES_INCLUDE=/usr/local/cuda/samples/common/inc/
CUDA_INCLUDE=/usr/local/cuda/lib64 
CUDA_LIB=/usr/local/cuda-$(CUDA_VERSION)/lib64

# NOTE: These options can be read off by running nvidia's (copyrighted) Makefile on some test examples
NVCC_FLAGS=-m64 -gencode arch=compute_30,code=sm_30 -use_fast_math

all:
	nvcc $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I. -I$(CUDA_SAMPLES_INCLUDE) -o main.o -c main.cu
	g++ -m64 -o gpuropolis main.o -L$(CUDA_LIB) -lcudart -lstdc++ -lm
	rm main.o
	
clean:
	rm -f gpuropolis src/*.o *.o
