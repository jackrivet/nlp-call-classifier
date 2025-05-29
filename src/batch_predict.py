import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import boto3
from io import StringIO
from datetime import datetime

bucket = 'your-bucket'
current_date = datetime.today().strftime("%Y-%m-%d")
input_key = f'Analysis/Voice/parsed_transcripts/Call Batch - {current_date}.csv'
output_key = f'Analysis/Voice/categorized_calls/Labeled Call Batch - {current_date}.csv'
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_pretrained_tokenizer")

model = DistilBertForSequenceClassification.from_pretrained("saved_pretrained_model")
model.eval()
s3 = boto3.client('s3')

try:
    response = s3.get_object(Bucket=bucket, Key=input_key)
    body = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    print("Loaded key: ", input_key)
except s3.exceptions.NoSuchKey:
    print("No file found for: ", current_date, input_key)
    df = pd.DataFrame()
  
if not df.empty:
    texts = df['transcript'].astype(str).tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encodings)
        predicted_class_ids = torch.argmax(outputs.logits, dim=1)
      
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    df["predicted_label"] = [id2label[i.item()] for i in predicted_class_ids]
  
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=output_key, Body=buffer.getvalue())
    print(f"Uploaded labeled calls to: {output_key}")
else:
    print("No data")
