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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = AutoModelForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased")

queries = []
paragraphs = []
labels = []

# Test dataset
# queries = ["Jag heter Max","Idag släpper KB tre språkmodeller."]
# paragraphs = [
#     "Verksamheten är uppdelad på ett antal laboratorier",
#     "Jättebra",
# ]
# labels = [0,1]


# Real Dataset, read from paragraphs.json bulk upload file
# Open the list of articles to read
paragraphsfile = open('paragraphs_test.json', 'r')
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

# print("Positive")
# print(len(queries), queries)
# print(len(paragraphs), paragraphs)
# print(len(labels), labels)

# Create as many negative examples as positive (negative: label = 0)
neg_count = pos_count
neg_queries = []
neg_paragraphs = []
neg_labels = []
for i in range(neg_count):
    random_index = randrange(neg_count)
    # Find a random query that is paired with another paragraph than its correct paragraph
    while (queries[random_index].partition(' ')[0] == queries[i].partition(' ')[0]): # find new pair if first word of query is the same
        random_index = randrange(neg_count)
    neg_queries.append(queries[random_index])
    neg_paragraphs.append(paragraphs[i])
    neg_labels.append(0) # these are all negative examples, that is these paragraphs are not relevant to the query


# print("Negative")
# print(len(neg_queries), neg_queries)
# print(len(neg_paragraphs), neg_paragraphs)
# print(len(neg_labels), neg_labels)

queries.extend(neg_queries)
paragraphs.extend(neg_paragraphs)
labels.extend(neg_labels)

# print("Both")
# print(len(queries), queries)
# print(len(paragraphs), paragraphs)
# print(len(labels), labels)




# Will pad the sequences up to the model max length
model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512)
# model_inputs = tokenizer(queries, paragraphs)

# print(model_inputs)

# decoded = tokenizer.decode(model_inputs["input_ids"][0])

# print(decoded)



dataset = WikiDataset(model_inputs, labels)
# print("dataset.encodings",dataset.encodings["input_ids"])

# print("dataset.labels",dataset.labels)


train_set_size = int(len(dataset) * 0.8)
val_set_size = len(dataset) - train_set_size
train_dataset, val_dataset = data.random_split(dataset, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(42))
# print("val_dataset", val_dataset.dataset)
# print("val_dataset", val_dataset.indices)
# print(len(val_dataset))
# print(len(train_dataset))

# print(val_dataset.labels)




# Training

# def compute_metrics(p):    
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     recall = recall_score(y_true=labels, y_pred=pred)
#     precision = precision_score(y_true=labels, y_pred=pred)
#     f1 = f1_score(y_true=labels, y_pred=pred)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir='./output',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    optim='adamw_torch',
    # seed=0,
    # evaluation_strategy="steps",
    # load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,           # evaluation dataset
    # compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()


torch.save(model, 'model_C_3_test')


saved_model = torch.load('model_C_3_test')


trainer = Trainer(
    model=saved_model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,           # evaluation dataset
    # compute_metrics=compute_metrics,
)

trainer.evaluate()

# from torch.utils.data import DataLoader
# from torch.optim import AdamW

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model.to(device)
# model.train()

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# valid_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     for batch in train_loader:
#         print("batch", batch)
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         # labels = torch.unsqueeze(labels, 1)
#         print("input_ids", len(input_ids), input_ids.shape)
#         print("labels", len(labels), labels.shape)
#         print("attention_mask", len(attention_mask), attention_mask.shape)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()

# model.eval()




