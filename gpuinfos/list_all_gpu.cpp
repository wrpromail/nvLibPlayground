#include <iostream>
#include <cuda_runtime.h>

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    std::cout << "Device " << device << ": " << device_prop.name << std::endl;
    std::cout << "  Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    std::cout << "  Total global memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Multiprocessor count: " << device_prop.multiProcessorCount << std::endl;
  }

  return 0;
}
