#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using Matrix = std::vector<std::vector<float>>;

Matrix load_matrix_from_json(const json& j_matrix) {
    size_t rows = j_matrix.size();
    size_t cols = j_matrix[0].size();
    Matrix matrix(rows, std::vector<float>(cols, 0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = j_matrix[i][j];
        }
    }
    return matrix;
}

int main() {
    std::ifstream weights_file("weights.json");
    json weights_json;
    weights_file >> weights_json;

    Matrix fc1_weight = load_matrix_from_json(weights_json["fc1_weight"]);
    Matrix fc1_bias = load_matrix_from_json(weights_json["fc1_bias"]);
    Matrix fc2_weight = load_matrix_from_json(weights_json["fc2_weight"]);
    Matrix fc2_bias = load_matrix_from_json(weights_json["fc2_bias"]);

    // 接下来，您可以使用这些权重和偏置矩阵进行推理计算

    return 0;
}