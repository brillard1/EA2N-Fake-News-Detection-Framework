import csv
import json
import random
import tensorflow as tf
import re
import sys

def read_jsonl(input_file):
    with tf.io.gfile.GFile(input_file, 'r') as f:
        return [json.loads(ln) for ln in f]

def combine_and_shuffle(jsonl_files, output_file):
    combined_data = []

    for jsonl_file, data_type in jsonl_files:
        print(jsonl_file)
        data = read_jsonl(jsonl_file)
        combined_data.extend([(item, data_type) for item in data])

    random.shuffle(combined_data)
    
    #test_size = int(0.2 * len(combined_data))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'content', 'label', 'type'])
        cnt1, cnt2 = 0,0
        for item, data_type in combined_data:
            data_type = 'test' if random.random() < 0.2 else 'training'
            id = item['id']
            content = item['question']['stem']
            content = str(content)
            
            # Checking for empty strings
            if content == 'nan':
              continue

            content = re.sub(r"[\n\r;@#!]|http[s]?://\S+|\.+", ' ', content)
            label = item['answerKey']
            if (label == 0):
              cnt1 += 1
            elif (label == 1):
              cnt2 += 1
            writer.writerow([id, content, label, data_type])
        print('real:', cnt1, 'fake:', cnt2)

# Provide the file paths and names
dataset = sys.argv[1]
train_jsonl_file = f'{dataset}/jsonl_data/train_rand_split.jsonl'
test_jsonl_file = f'{dataset}/jsonl_data/test_rand_split.jsonl'
dev_jsonl_file = f'{dataset}/jsonl_data/dev_rand_split.jsonl'
output_csv_file = f'{dataset}/{dataset}.csv'

# Specify the JSONL files and their corresponding data types
#jsonl_files = [(train_jsonl_file, 'training'), (test_jsonl_file, 'test'), (dev_jsonl_file, 'test')]
jsonl_files = [(train_jsonl_file, 'training')]
print('Step 1. Completed.')

print(jsonl_files)

# Generate the combined and shuffled CSV file
combine_and_shuffle(jsonl_files, output_csv_file)
print('Step 2. Completed.')