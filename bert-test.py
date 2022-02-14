import numpy as np
import datasets
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification


imdb = load_dataset("imdb")

# imdb = dataset.shard(num_shards=40, index=3)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_imdb = imdb.map(preprocess_function, batched=True)


print("raw_datasets", imdb)

print(imdb["train"][0])


# print("text:", tokenized_datasets["train"][0]["text"])
# print("label:", tokenized_datasets["train"][0]["label"])

# print("text:", tokenized_datasets["train"][1]["text"])
# print("label:", tokenized_datasets["train"][1]["label"])

# print("text:", tokenized_datasets["train"][2]["text"])
# print("label:", tokenized_datasets["train"][2]["label"])

print("tokenized_datasets",tokenized_imdb)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# full_train_dataset = tokenized_datasets["train"]
# full_eval_dataset = tokenized_datasets["test"]


# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


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


