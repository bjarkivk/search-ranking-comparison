import json
from random import randrange
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, AutoModelForSequenceClassification, TextClassificationPipeline
from transformers import TrainingArguments, Trainer
import torch
import torch.utils.data as data
import numpy as np


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = torch.load('model_B_10k',map_location ='cpu') # When we only have cpu



queries = ["Jag heter Max",
    "Idag släpper KB tre språkmodeller.",
    "Nordlys (musikalbum) Låtförteckning"
]
paragraphs = [
    "Namnet er bra",
    "Jättebra",
    "Text: Carmen Elise Espenæs Musik: Midnattsol",
]
model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512, return_tensors="pt")


# forward pass
outputs = model(**model_inputs)
predictions = outputs.logits.argmax(-1)
print(model_inputs)
print("outputs", outputs)
print("predictions", predictions)
print(outputs.logits.softmax(dim=-1).tolist())


