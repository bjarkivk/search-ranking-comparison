import json
from random import randrange
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import datasets
import torch
import torch.utils.data as data
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from WikiDataset import WikiDataset
import time
start = time.time()


### This file was just a test, it was not used ###


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = torch.load('model_D_100k',map_location ='cpu') # When we only have cpu

queries = []
paragraphs = []
labels = []



# Real Dataset, read from paragraphs.json bulk upload file
# Open the list of articles to read
paragraphsfile = open('paragraphs_test2.json', 'r')
lines = paragraphsfile.readlines()

# Create positive examples (label = 1)
for index, line in enumerate(lines):
    line = line.replace('\n', '')
    if(line != '{"index":{}}'): # We discard the empty lines
        y = json.loads(line)
        q = y["id1"] + " " + y["id2"] + " " + y["id3"] + " " + y["id4"] + " " + y["id5"] # Create a query that is all ids concatinated together with space between
        query = q.strip() # Remove spaces at the end
        queries.append(query)
        paragraphs.append(y["paragraph"])
        labels.append(1) # these are all positive examples, that is these paragraphs are relevant to the query

pos_count = len(queries)

# Create as many negative examples as positive (negative: label = 0)
neg_count = pos_count
neg_queries = []
neg_paragraphs = []
neg_labels = []
for i in range(neg_count):
    random_index = randrange(neg_count)
    # Find a random paragraph that is paired with another query than its correct query
    while (queries[random_index] == queries[i]): 
        random_index = randrange(neg_count)
    neg_queries.append(queries[i])
    neg_paragraphs.append(paragraphs[random_index])
    neg_labels.append(0) # these are all negative examples, that is these paragraphs are not relevant to the query


queries.extend(neg_queries)
paragraphs.extend(neg_paragraphs)
labels.extend(neg_labels)





# Will pad the sequences up to the model max length
model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512, return_tensors="pt")



# dataset = WikiDataset(model_inputs, labels)





# forward pass
outputs = model(**model_inputs)
predictions = outputs.logits.argmax(-1)
print(model_inputs)
print("outputs", outputs)
print("predictions", predictions)
print(outputs.logits.softmax(dim=-1).tolist())

correct_pred = 0
pred = predictions.tolist()
print(pred, type(pred))
print(labels, type(labels))
for i, value in enumerate(pred):
    if (value == labels[i]):
        correct_pred += 1
print("correct_pred", correct_pred)
print("percentage", correct_pred/len(pred))

end = time.time()
print("time elapsed", end - start)


# train_set_size = int(len(dataset) * 0.80)
# val_set_size = len(dataset) - train_set_size
# train_dataset, val_dataset = data.random_split(dataset, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(42))





# # Evaluation

# # saved_model = torch.load('model_B_10k')
# saved_model = torch.load('model_B_10k',map_location ='cpu') # When we only have cpu

# training_args = TrainingArguments(
#     output_dir='./output',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=64,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=10,
#     optim='adamw_torch',
#     # seed=0,
#     # evaluation_strategy="steps",
#     # load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=saved_model,                   # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset,            # evaluation dataset
# )

# metrics=trainer.evaluate()
# print(metrics)

