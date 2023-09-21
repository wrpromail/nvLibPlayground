import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 准备数据
# 20newsgroups 数据集 https://huggingface.co/datasets/SetFit/20_newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# 使用 LabelEncoder 处理标签
le = LabelEncoder()
train_labels = le.fit_transform(newsgroups_train.target)
test_labels = le.transform(newsgroups_test.target)
# 分割训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(newsgroups_train.data, train_labels, test_size=.2)
# 2. 词条化
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(newsgroups_test.data, truncation=True, padding=True)


class NewsGroupsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsGroupsDataset(train_encodings, train_labels)
val_dataset = NewsGroupsDataset(val_encodings, val_labels)
test_dataset = NewsGroupsDataset(test_encodings, test_labels)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(newsgroups_train.target)))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    # 需要根据自己的显存大小设置
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
