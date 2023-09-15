
#### 学习范围
cudnnpg cuDNN 库
gpuedge 对比CPU、GPU下进行相同运算
gpuinfos 获取 GPU 信息
ncclpg 验证多GPU下数据通信
precision 验证不同精度
tensorpg 对比使用 CPU\CUDA core\tensor core 下进行相同运算




#### 运行方式
使用 docker 容器开启测试环境
```
# 使用 nvidia-smi 查看宿主机CUDA 版本
# 使用a.b.c 中 ab版本与宿主机 CUDA 版本一致的镜像，注意使用 devel 而非 runtime 镜像
docker run -it --rm --gpus all -v /root/wangrui/libpg:/app cs-ai.tencentcloudcr.com/triton/cuda:11.4.3-cudnn8-devel-ubuntu20.04 /bin/bash
```


```
# 带上 cuBLAS 库进行编译
nvcc -o output_binary source_code.cpp -lcublas
# 带上 cuDNN 库进行编译
nvcc -o output_binary source_code.cpp -lcudnn
```
