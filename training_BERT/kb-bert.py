import json
from random import randrange
# from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
# from transformers import TrainingArguments, Trainer
# import datasets
# import torch

# class WikiDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
# model = AutoModelForMaskedLM.from_pretrained("KB/bert-base-swedish-cased")

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
article_file = open('../training_BERT/paragraphs.json', 'r')
lines = article_file.readlines()

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

print("Positive")
print(len(queries), queries)
print(len(paragraphs), paragraphs)
print(len(labels), labels)

# Create negative examples (label = 0)
neg_count = pos_count
neg_queries = []
neg_paragraphs = []
neg_labels = []
for i in range(neg_count):
    random_index = randrange(neg_count)
    # Find a paragraph that is paired with another query than its correct query
    while (queries[random_index] == queries[i]): 
        random_index = randrange(neg_count)
    neg_queries.append(queries[i])
    neg_paragraphs.append(paragraphs[random_index])
    neg_labels.append(0) # these are all negative examples, that is these paragraphs are not relevant to the query


print("Negative")
print(len(neg_queries), neg_queries)
print(len(neg_paragraphs), neg_paragraphs)
print(len(neg_labels), neg_labels)

queries.extend(neg_queries)
paragraphs.extend(neg_paragraphs)
labels.extend(neg_labels)

print("Both")
print(len(queries), queries)
print(len(paragraphs), paragraphs)
print(len(labels), labels)




# Will pad the sequences up to the model max length
# model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512)
# model_inputs = tokenizer(queries, paragraphs)



# print(model_inputs)

# decoded = tokenizer.decode(model_inputs["input_ids"][0])

# print(decoded)


# train_dataset = WikiDataset(model_inputs, labels)

# print("train_dataset", train_dataset.encodings, train_dataset.labels)



# Training

# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_imdb["train"],
#     eval_dataset=tokenized_imdb["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()




