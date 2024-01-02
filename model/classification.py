import config
import json
from transformers import pipeline
from tqdm import tqdm

path_to_json = '/Users/ansixu/Personality_Project/text_train_data.json'
# Load the JSON file
with open(path_to_json, 'r') as file:
    text_data = json.load(file)
# Initialize a text classification pipeline with a pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the topic labels (you can modify these to suit your topics)
labels = ["topic1", "topic2", "topic3", "topic4", "topic5", 
          "topic6", "topic7", "topic8", "topic9", "topic10"]
try:
# Classify each text and add the result to the JSON data
    for filename in tqdm(text_data.keys(),desc='Classifying Text'):
        text =text_data[filename]
        result = classifier(text,labels)
        top_class = result['labels'][0]
        text_data[filename] ={'text':text,'class':top_class}   
except:
    print(filename)
# Save the updated data to a new JSON file
with open('updated_classified_train_text_data.json', 'w') as updated_file:
    json.dump(text_data, updated_file)