import numpy as np

def quantize_weights(weights, dtype=np.int8):
    # 将权重缩放到 [0, 255] 范围
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    scale = (max_weight - min_weight) / 255
    quantized_weights = (weights - min_weight) / scale
    # 四舍五入为整数并转换为目标数据类型
    quantized_weights = np.round(quantized_weights).astype(dtype)
    # scale, min_weight 用于将量化后的权重还原为 fp32
    return quantized_weights, scale, min_weight