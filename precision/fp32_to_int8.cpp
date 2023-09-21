//
// Created by wangrui on 2023/9/21.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

using Martix = std::vector<std::vector<float>>;

std::vector<std::vector<int8_t>> quantize(const Martix &weights) {
    float min_weight = std::numeric_limits<float>::max();
    float max_weight = std::numeric_limits<float>::min();
    for (const auto& row : weights) {
        for (const auto& value : row) {
            min_weight = std::min(min_weight, value);
            max_weight = std::max(max_weight, value);
        }
    }

    float scale = (max_weight - min_weight) / 255.0f;
    // weights.size() 指量化后的矩阵行数与原矩阵相同
    // weights[0].size() 指量化后的矩阵列数与原矩阵相同
    std::vector<std::vector<int8_t>> quantized_weights(weights.size(), std::vector<int8_t>(weights[0].size()));
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            quantized_weights[i][j] = std::round((weights[i][j] - min_weight) / scale);
        }
    }
    return quantized_weights;
}

int main() {
    Martix weights = {
            {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6},
            {0.7, 0.8, 0.9}
    };
    auto quantized_weights = quantize(weights);
    for (const auto &row : quantized_weights) {
        for (int8_t value : row) {
            std::cout << static_cast<int>(value) << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}