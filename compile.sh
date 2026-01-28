#!/bin/bash

if [ ! -d "./elf" ]; then
    mkdir ./elf
fi

c++ -std=c++20 -I./include src/timer.cpp src/single_thread.cpp -o elf/single_thread.elf
c++ -std=c++20 -I./include src/timer.cpp src/multi_thread.cpp -o elf/multi_thread.elf
nvcc -I./include src/timer.cpp src/naive_cuda.cu -o elf/naive_cuda.elf
nvcc -I./include src/timer.cpp src/optimize.cu -o elf/optimize.elf
