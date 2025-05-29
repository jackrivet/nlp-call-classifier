import json
import boto3
import csv
import os
from io import StringIO
from datetime import datetime
import botocore
import urllib.parse

s3 = boto3.client('s3')

def lambda_handler(event, context):
    print("Event:", json.dumps(event))
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    response = s3.get_object(Bucket=bucket, Key=key)
    content = json.loads(response['Body'].read().decode('utf-8'))

    contact_id = content.get("CustomerMetadata", {}).get("ContactId", "unknown")
    transcript_lines = []

    for entry in content.get('Transcript', []):
        if entry.get('ParticipantId') == 'CUSTOMER':
            transcript_lines.append(entry.get('Content', ''))

    full_transcript = ' '.join(transcript_lines).strip()

    if not full_transcript:
        return {'statusCode': 200, 'body': 'No customer content found.'}

    date_string = datetime.now().strftime("%Y-%m-%d")
    base_prefix = "/".join(key.split("/")[:2])
    output_key = f"{base_prefix}/parsed_transcripts/Call Batch - {date_string}.csv"

    try:
        existing_obj = s3.get_object(Bucket=bucket, Key=output_key)
        csv_body = existing_obj['Body'].read().decode('utf-8')
        rows = list(csv.reader(StringIO(csv_body)))

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            rows = [['contact_id', 'transcript', 'source']]  # <-- updated header
        else:
            raise e

    rows.append([contact_id, full_transcript, key])  # <-- updated row
    csv_buffer = StringIO()

    writer = csv.writer(csv_buffer)
    writer.writerows(rows)

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=csv_buffer.getvalue(),
    )
    
    print("Decoded Key:", key, output_key, len(full_transcript), bucket)
    return {
        'statusCode': 200,
        'body': f'Transcript appended to {output_key}'
    }
