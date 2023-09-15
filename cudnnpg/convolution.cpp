#include <iostream>
#include <vector>
#include <cudnn.h>

int main() {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  // 输入特征图（NCHW 格式）
  int N = 1, C = 3, H = 8, W = 8;
  float *input_data;
  cudaMallocManaged(&input_data, N * C * H * W * sizeof(float));

  cudnnTensorDescriptor_t input_descriptor;
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);

  // 卷积核（KCRS 格式）
  int K = 1, R = 3, S = 3;
  float *kernel_data;
  cudaMallocManaged(&kernel_data, K * C * R * S * sizeof(float));

  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S);

  // 卷积操作
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnCreateConvolutionDescriptor(&convolution_descriptor);
  cudnnSetConvolution2dDescriptor(convolution_descriptor, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  // 输出特征图
  int out_N, out_C, out_H, out_W;
  cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor, &out_N, &out_C, &out_H, &out_W);

  float *output_data;
  cudaMallocManaged(&output_data, out_N * out_C * out_H * out_W * sizeof(float));

  cudnnTensorDescriptor_t output_descriptor;
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_N, out_C, out_H, out_W);

  // 卷积前向传播
  float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_perf;
  int returned_algo_count;
  cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, 1, &returned_algo_count, &convolution_algorithm_perf);
  cudnnConvolutionFwdAlgo_t convolution_algorithm = convolution_algorithm_perf.algo;

  size_t workspace_size;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size);


  void *workspace;
  cudaMalloc(&workspace, workspace_size);

  cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input_data, kernel_descriptor, kernel_data, convolution_descriptor, convolution_algorithm, workspace, workspace_size, &beta, output_descriptor, output_data);

  // 清理资源
  cudaFree(input_data);
  cudaFree(kernel_data);
  cudaFree(output_data);
  cudaFree(workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

  return 0;
}
