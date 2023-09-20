import pickle
from train_code import MLP
import torch
import numpy as np

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

def get_input_vector(input_string: str):
    return loaded_vectorizer.transform([input_string]).toarray()

input_size = 13656
hidden_size = 64
num_classes = 2
loaded_model = MLP(input_size, hidden_size, num_classes)
weights = torch.load('model.pth')
loaded_model.load_state_dict(weights)
params = loaded_model.state_dict()
fc1_weight = params['fc1.weight']
fc1_bias = params['fc1.bias']
fc2_weight = params['fc2.weight']
fc2_bias = params['fc2.bias']

def custom_inference(input_matrix, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    # 输入矩阵与 fc1_weight 相乘并加上 fc1_bias
    fc1_output = np.dot(input_matrix, fc1_weight.T) + fc1_bias
    # 应用 ReLU 激活函数
    relu_output = np.maximum(0, fc1_output)
    # 将 ReLU 输出与 fc2_weight 相乘并加上 fc2_bias
    fc2_output = np.dot(relu_output, fc2_weight.T) + fc2_bias
    # 获取预测结果
    prediction = np.argmax(fc2_output, axis=1)
    return prediction


if __name__ == "__main__":
    text_vector = get_input_vector("I feel good")
    input_matrix = text_vector  # 假设已经将文本转换为特征向量
    input_matrix_np = input_matrix.astype(np.float32)  # 确保输入矩阵是 float32 类型的 Numpy 数组
    # 将 PyTorch 张量转换为 Numpy 数组
    fc1_weight_np = fc1_weight.numpy()
    fc1_bias_np = fc1_bias.numpy()
    fc2_weight_np = fc2_weight.numpy()
    fc2_bias_np = fc2_bias.numpy()

    prediction = custom_inference(input_matrix_np, fc1_weight_np, fc1_bias_np, fc2_weight_np, fc2_bias_np)
    print(f"Prediction: {prediction}")  # 输出 1 表示代码，0 表示非代码