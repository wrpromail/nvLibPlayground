cmake_minimum_required(VERSION 3.18.2)
#project(LearnCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(nlohmann_json REQUIRED)

add_executable(LearnCUDA fp32_to_int8.cpp
        ../simple_classification/cpu_implementation/read_weights.cpp
        ../simple_classification/cpu_implementation/json.hpp)


#add_subdirectory(cutlass/)
target_link_libraries(LearnCUDA PRIVATE nlohmann_json::nlohmann_json)


#set_target_properties(LearnCUDA PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)