# CUDA_VERSION=7.0
CUDA_VERSION=8.0
# CUDA_VERSION=5.5

CUDA_SAMPLES_INCLUDE=/usr/local/cuda/samples/common/inc/
CUDA_INCLUDE=/usr/local/cuda/lib64 
CUDA_LIB=/usr/local/cuda-$(CUDA_VERSION)/lib64

# NOTE: These options can be read off by running nvidia's (copyrighted) Makefile on some test examples
NVCC_FLAGS=-m64 -gencode arch=compute_30,code=sm_30 -use_fast_math
# -use_fast_math forces everything to compile to __exp(x) versions, instead of slower and more accurate exp(x) versions...

all: main.o
	g++ -m64 -o gpumcmc main.o -L$(CUDA_LIB) -lcudart -lstdc++ -lm
	rm main.o
	 
main.o: src/*.cu

	nvcc $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I. -I$(CUDA_SAMPLES_INCLUDE) -o main.o -c src/main.cu 
	
clean:
	rm -f gpumcmc src/*.o *.o
