#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
  int M = 1024 * 2;
  int N = 1024 * 2;
  int K = 1024 * 2;

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 1.0f);
  std::vector<float> C_cpu(M * N, 0.0f);
  std::vector<float> C_gpu(M * N, 0.0f);

  // CPU 矩阵乘法
  auto start_cpu = std::chrono::high_resolution_clock::now();
  matrix_mult_cpu(A, B, C_cpu, M, N, K);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  // GPU 矩阵乘法 - cublasSgemm
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

  auto start_gpu_sgemm = std::chrono::high_resolution_clock::now();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
  auto end_gpu_sgemm = std::chrono::high_resolution_clock::now();

  cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // GPU 矩阵乘法 - cublasGemmEx (Tensor Cores)
  half *d_A_fp16, *d_B_fp16, *d_C_fp16;
  cudaMalloc((void**)&d_A_fp16, M * K * sizeof(half));
  cudaMalloc((void**)&d_B_fp16, K * N * sizeof(half));
  cudaMalloc((void**)&d_C_fp16, M * N * sizeof(half));

  // 将输入矩阵转换为 FP16
  std::vector<half> A_fp16(M * K);
  std::vector<half> B_fp16(K * N);
  for (int i = 0; i < M * K; ++i) {
    A_fp16[i] = __float2half(A[i]);
  }
  for (int i = 0; i < K * N; ++i) {
    B_fp16[i] = __float2half(B[i]);
  }

  cudaMemcpy(d_A_fp16, A_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_fp16, B_fp16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

  auto start_gpu_gemmex = std::chrono::high_resolution_clock::now();
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A_fp16, CUDA_R_16F, M, d_B_fp16, CUDA_R_16F, K, &beta, d_C_fp16, CUDA_R_16F, M, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
  auto end_gpu_gemmex = std::chrono::high_resolution_clock::now();

  // 将 FP16 结果转换回 FP32
  std::vector<half> C_gpu_fp16(M * N);
  cudaMemcpy(C_gpu_fp16.data(), d_C_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost);
  for (int i = 0; i < M * N; ++i) {
    C_gpu[i] = __half2float(C_gpu_fp16[i]);
  }

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_A_fp16);
  cudaFree(d_B_fp16);
  cudaFree(d_C_fp16);

  auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
  auto gpu_sgemm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_sgemm - start_gpu_sgemm).count();
  auto gpu_gemmex_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_gemmex - start_gpu_gemmex).count();

  std::cout << "CPU time: " << cpu_duration << " ms" << std::endl;
  std::cout << "GPU time (cublasSgemm): " << gpu_sgemm_duration << " ms" << std::endl;
  std::cout << "GPU time (cublasGemmEx): " << gpu_gemmex_duration << " ms" << std::endl;

  return 0;
}
