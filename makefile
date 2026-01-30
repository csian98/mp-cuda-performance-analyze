CXX := c++
NVCC := nvcc
AR := ar
LD := ld

INC_DIR := include /opt/cuda/include
INC_DIRS := $(INC_DIR)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
CXXFLAGS := -std=c++20 $(INC_FLAGS)
NVCCFLAGS := $(INC_FLAGS) -lcublas
SHD_NVCCFLAGS := -Xcompiler -fPIC -D_TARGET_OS=OS_LINUX
SHD_LDFLAGS := -Xcompiler -shared

elf/single_thread.elf: src/timer.cpp src/single_thread.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

elf/multi_thread.elf: src/timer.cpp src/multi_thread.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

elf/naive_cuda.elf: src/timer.cpp src/naive_cuda.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

elf/optimize_cuda.elf: src/timer.cpp src/optimize_cuda.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

elf/cuBLAS.elf: src/timer.cpp src/cuBLAS.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

elf/convolution.elf: src/timer.cpp src/convolution.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

lib/libcuda2py.so: src/cuda2py.cu
	$(NVCC) $(SHD_NVCCFLAGS) $(SHD_LDFLAGS) $^ -o $@

all: elf/single_thread.elf elf/multi_thread.elf elf/naive_cuda.elf elf/optimize_cuda.elf elf/cuBLAS.elf elf/convolution.elf lib/libcuda2py.so
