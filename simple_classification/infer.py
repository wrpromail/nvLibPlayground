import torch
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


def predict(model, text):
    # 将文本转换为特征向量
    text_vector = vectorizer.transform([text]).toarray()
    # 将数据转换为 PyTorch 张量
    text_tensor = torch.tensor(text_vector, dtype=torch.float32)
    # 前向传播
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 示例


if __name__ == "__main__":
    mys = "my mood is really down"
    prediction = predict(model,mys)
    print(prediction)
# 0
    mys = "I feel good"
    prediction = predict(model,mys)
    print(prediction)
# 1