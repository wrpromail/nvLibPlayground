import torch
import torch.nn as nn

# input_size 和训练数据集有关
input_size = 13656
hidden_size = 64
num_classes = 2
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 创建一个新的实例
loaded_model = MLP(input_size, hidden_size, num_classes)

# 加载权重文件
weights = torch.load('model.pth')

# 将权重应用于神经网络实例
loaded_model.load_state_dict(weights)

# 读取参数
params = loaded_model.state_dict()
fc1_weight = params['fc1.weight']
fc1_bias = params['fc1.bias']
fc2_weight = params['fc2.weight']
fc2_bias = params['fc2.bias']