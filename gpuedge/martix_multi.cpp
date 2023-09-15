#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CPU 矩阵乘法
void matrix_mult_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

int main() {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 1.0f);
  std::vector<float> C_cpu(M * N, 0.0f);
  std::vector<float> C_gpu(M * N, 0.0f);

  // CPU 矩阵乘法
  auto start_cpu = std::chrono::high_resolution_clock::now();
  matrix_mult_cpu(A, B, C_cpu, M, N, K);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  // GPU 矩阵乘法
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, M * K * sizeof(float));
  cudaMalloc((void**)&d_B, K * N * sizeof(float));
  cudaMalloc((void**)&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  auto start_gpu = std::chrono::high_resolution_clock::now();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
  auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();

  std::cout << "CPU time: " << cpu_duration << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_duration << " ms" << std::endl;

  return 0;
}
