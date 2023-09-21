from transformers import pipeline
from transformers import BertTokenizerFast, BertForSequenceClassification

model_path = './saved_model'
model = BertForSequenceClassification.from_pretrained(model_path)
# 保存目录下只有模型而没有 tokenizer，所以需要重新加载
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

text = 'This is a test message.'
result = classifier(text)
print(result)