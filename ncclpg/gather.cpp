#include <iostream>
#include <vector>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 矩阵乘法（GPU）
void matrix_mult_gpu(const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int device) {
  cudaSetDevice(device);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

  cublasDestroy(handle);
}

int main() {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 1.0f);
  std::vector<float> C(M * N, 0.0f);

  float *d_A, *d_B, *d_C;
  cudaSetDevice(0);
  cudaMalloc((void**)&d_A, M * K * sizeof(float));
  cudaMalloc((void**)&d_B, K * N * sizeof(float));
  cudaMalloc((void**)&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  matrix_mult_gpu(d_A, d_B, d_C, M, N, K, 0);

  // 使用 NCCL 将结果从 GPU0 发送到 GPU1
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  int num_gpus = 2;

  ncclGetUniqueId(&nccl_id);
  ncclCommInitRank(&nccl_comm, num_gpus, nccl_id, 0);
  ncclAllReduce(d_C, d_C, M * N, ncclFloat, ncclSum, nccl_comm, cudaStreamDefault);

  cudaSetDevice(1);
  float *d_C_gpu1;
  cudaMalloc((void**)&d_C_gpu1, M * N * sizeof(float));
  cudaMemcpy(d_C_gpu1, d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice);

  ncclCommDestroy(nccl_comm);

  // 在 GPU1 上进行后续计算（例如，将结果与另一个矩阵相乘）

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_gpu1);

  return 0;
}
