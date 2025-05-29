import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

df = pd.read_csv("Training Data for Call Categorization")
df.dropna(subset=['transcript', 'label'], inplace=True)

unique_labels = sorted(df['label'].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label_id'] = df['label'].map(label2id)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['transcript'], df['label_id'], test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer/")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

class CallDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

  
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

  
    def __len__(self):
        return len(self.labels)


train_dataset = CallDataset(train_encodings, list(train_labels))
test_dataset = CallDataset(test_encodings, list(test_labels))

model = DistilBertForSequenceClassification.from_pretrained(
    "model/",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="sagemaker_results2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=6,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_total_limit=1,
    greater_is_better=False,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
metrics = trainer.evaluate()
print("Evaluation Metrics: ", metrics)

model.save_pretrained("saved_pretrained_model")
tokenizer.save_pretrained("saved_pretrained_tokenizer")
