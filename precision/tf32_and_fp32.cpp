#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    int N = 4096;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 分配和初始化输入矩阵 A 和 B
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    // 创建 CUDA 和 cuBLAS 句柄
    cudaStream_t stream;
    cublasHandle_t handle;
    cudaStreamCreate(&stream);
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // 分配 GPU 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // 将输入数据复制到 GPU
    cudaMemcpyAsync(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 在 FP32 精度下执行矩阵乘法
    auto start_fp32 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaStreamSynchronize(stream);
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_fp32 = end_fp32 - start_fp32;

    // 将结果复制回主机内存
    cudaMemcpyAsync(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 在 TF32 精度下执行矩阵乘法（需要 Ampere 架构 GPU，如 A100）
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    auto start_tf32 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaStreamSynchronize(stream);
    auto end_tf32 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tf32 = end_tf32 - start_tf32;

    // 释放 GPU 内存和句柄
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    // 打印计算时间
    std::cout << "FP32 elapsed time: " << elapsed_fp32.count() << " seconds" << std::endl;
    std::cout << "TF32 elapsed time: " << elapsed_tf32.count() << " seconds" << std::endl;

    return 0;
}
