from transformers import BertForSequenceClassification

loaded_model = BertForSequenceClassification.from_pretrained('./saved_model')
for layer_index, layer in enumerate(loaded_model.bert.encoder.layer):
    print(f'Layer {layer_index} weight shape: {layer.attention.self.query.weight.shape}')
    print(f'Layer {layer_index} bias shape: {layer.attention.self.query.bias.shape}')

layers = loaded_model.bert.encoder.layer
layer1 = layers[0]
print(f'Layer 1 weight shape: {layer1.attention.self.query.weight.shape}')

state_dict = layer1.state_dict()

# 根据 base_bert_config.json 文件以及上述代码可知模型的分布情况
# 12隐藏层
# 12注意力头
# 每头注意力维度为 64
# hidden_size / heads_count = 64
# 总参数量 110 M
