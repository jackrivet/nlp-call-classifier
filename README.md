# Amazon Connect NLP Call Classifier
An end-to-end NLP classification pipeline built on AWS to analyze call transcripts from Amazon Connect using a DistilBERT model deployed via SageMaker. The system automates transcript processing, classification, and result delivery using AWS Lambda and S3.

## Table of Contents
- [Overview](#pipeline-overview)
- [Model](#model)
- [Training](#training)
- [Inference](#inference)
- [Tech Stack](#tech-stack)
- [Deployment](#deployment)
- [Author](#author)

<img src="https://github.com/jackrivet/nlp-call-classifier/blob/main/Process-Diagram.png" alt="Process Diagram" width="700"/>

## Pipeline Overview
- **Amazon Connect** stores customer call transcripts (via Contact Lens) as JSON files in S3.
- **S3 Event Triggers** a **Lambda** function on new file creation.
- **Lambda** extracts and appends transcript text to a master CSV file in S3.
- **SageMaker** loads the CSV, runs batch predictions using a **fine-tuned DistilBERT** model.
- Predictions are stored in a final labeled CSV in S3, ready for reporting or further analysis.

## Model
- **Model:** `DistilBERT` via `HuggingFace Transformers`
- **Training Framework:** AWS SageMaker
- **Task:** Multi-class text classification
- **Use Case:** Categorize call types

## Training
The model is fine-tuned using Hugging Face Transformers and AWS SageMaker. The training script performs the following steps:
-	Loads and cleans a labeled CSV of manually tagged call transcripts.
- Maps class labels to numeric IDs and splits the data into training and testing sets.
-	Tokenizes the transcripts using a DistilBERT tokenizer.
- Wraps the encoded data into PyTorch-compatible datasets.
- Fine-tunes a DistilBERT model using the Hugging Face Trainer API.
- Saves the trained model and tokenizer locally for use in SageMaker batch inference.
  
Training code is located in src/train_model.py

Training parameters include:
- Model: DistilBERT (base uncased)
-	Loss: Cross-entropy (multi-class)
-	Epochs: 5
- Optimizer: AdamW with weight decay
- Evaluation strategy: On each epoch using validation loss
  
After training, the model is saved and reused during batch inference to classify future call transcripts uploaded to S3.

## Inference

Once trained, the model is deployed in a serverless batch prediction workflow. The batch prediction script:
- Loads the most recent CSV of parsed transcripts from an S3 bucket.
- Tokenizes the transcripts using the saved DistilBERT tokenizer.
- Applies the fine-tuned model to generate predicted labels.
- Maps numeric predictions back to human-readable labels using model.config.id2label.
- Uploads a labeled CSV file back to S3 for reporting and downstream analysis.
 
Batch prediction script: src/batch_predict.py

The script is designed to be invoked via AWS Lambda or scheduled jobs, making it easy to automate classification as new call data becomes available in S3.

## Tech Stack
- Python (pandas, boto3 and others)
- Amazon SageMaker
- AWS Lambda
- Amazon S3
- DistilBERT via Hugging Face

## Deployment
- Configure Amazon Connect Contact Lens to store call transcripts as JSON files in an S3 Bucket.
- Deploy `convert-jsons-to-csv-for-sagemaker.py` in Lambda.
- Compile appropriate volume of training data by manually labeling transcripts with desired categories.
- Train model with `model_training.py` and save locally in SageMaker.
- Use `batch_predict.py` to apply labels to new transcripts and store back in S3.
- Monitor labeled transcripts in S3 or query via Athena.
- Configure downstream reporting.
  
## Author
For questions or collaboration opportunities, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/jackrivet/).

Notes: Amount of training data required to achieve accurate results scales with the desired number of categories.
