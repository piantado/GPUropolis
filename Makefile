
CUDA_SAMPLES_INCLUDE=/home/piantado/Build/NVIDIA_CUDA-5.0_Samples/common/inc
CUDA_INCLUDE=/usr/local/cuda-5.0/include

# NOTE: These options can be read off by running nvidia's (copyrighted) Makefile on some test examples
NVCC_FLAGS=-m64 -gencode arch=compute_30,code=sm_30

all: main.o
	# To see registers etc:
	# --ptxas-options=-v
	g++ -m64 -o gpumcmc main.o -L/usr/local/cuda-5.0/lib64 -lcudart

	
main.o: src/*.cu

	/usr/local/cuda-5.0/bin/nvcc $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I. -I$(CUDA_SAMPLES_INCLUDE) -o main.o -c src/main.cu 
	
clean:
	rm -f gpumcmc src/*.o *.o
