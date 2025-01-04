import os
import json
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import sys

# Set the path to the data directory
data_directory = sys.argv[1]

# Set the stopwords for question_concept extraction
stop_words = set(stopwords.words('english'))

# Create an empty list to store the extracted data
output_data = []
cnt_1 = 0
cnt_0 = 0

# Iterate over the folders in the root directory
for dirs in os.listdir(data_directory):
    for folder in os.listdir(os.path.join(data_directory,dirs)):
        folder_path = os.path.join(data_directory, dirs, folder)
        json_file_path = os.path.join(folder_path, 'news content.json')
        # Check if the JSON file exists in the folder
        if os.path.isfile(json_file_path):
            # Load the JSON file
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                # Extract the required attributes
                folder_id = folder
                title = data['title']
                text = data['text']
                final = title + ' ' + text
                if final in [' ','']:
                    continue

                # Handling Oversampling
                if 4852 + 1123 <= cnt_0 and dirs.lower() =='real':
                  break
                
                # Extract question_concept from title
                tokens = word_tokenize(title)
                question_concept = [token.lower() for token in tokens if token.lower() not in stop_words]

                # Determine the answerKey based on the folder
                answer_key = 1 if 'fake' in dirs.lower() else 0
                cnt_1 += 1 if answer_key == 1 else 0
                cnt_0 += 1 if answer_key == 0 else 0

                # Create the dictionary for the output data
                output_item = {
                    'answerKey': answer_key,
                    'id': folder_id,
                    'question': {
                        'question_concept': question_concept,
                        'choices': [{'label': 0, 'text': 'Real'}, {'label': 1, 'text': 'Fake'}],
                        'stem': final
                    }
                }

                # Append the output item to the list
                output_data.append(output_item)
                

data_set = sys.argv[2]
# Set the path for the output JSON file
train_output = f'{data_set}_train_rand_split.jsonl'
dev_output = f'{data_set}_dev_rand_split.jsonl'
test_output = f'{data_set}_test_rand_split.jsonl'

# Split the data into train, dev and test with even class distribution
train_data, test_data = train_test_split(output_data, test_size=0.2, random_state=42, stratify=[item['answerKey'] for item in output_data])
dev_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=[item['answerKey'] for item in test_data])

# Write the train data to the JSON file
with open(train_output, 'w') as json_output:
    for item in train_data:
        json_output.write(json.dumps(item))
        json_output.write('\n')

# Write the dev data to the JSON file
with open(dev_output, 'w') as json_output:
    for item in dev_data:
        json_output.write(json.dumps(item))
        json_output.write('\n')

# Write the test data to the JSON file
with open(test_output, 'w') as json_output:
    for item in test_data:
        json_output.write(json.dumps(item))
        json_output.write('\n')

print(f'Fake samples: {cnt_1}, Real samples: {cnt_0}')
print('Execution Complete..!')
