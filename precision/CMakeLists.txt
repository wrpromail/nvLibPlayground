cmake_minimum_required(VERSION 3.18.2)
#project(LearnCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_ARCHITECTURES 80)


add_executable(LearnCUDA fp32_to_int8.cpp)


#add_subdirectory(cutlass/)


#set_target_properties(LearnCUDA PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)