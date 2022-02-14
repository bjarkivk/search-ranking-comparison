from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from transformers import TrainingArguments, Trainer
import datasets
import torch

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = AutoModelForMaskedLM.from_pretrained("KB/bert-base-swedish-cased")


# sequences = ["Idag släpper KB tre språkmodeller.", "Jag heter Hans."]

batch_sentences = ["Jag heter Max","Idag släpper KB tre språkmodeller."]
batch_of_second_sentences = [
    "Verksamheten är uppdelad på ett antal laboratorier",
    "Jättebra",
]
labels = [0,1]


# Will pad the sequences up to the model max length
# model_inputs = tokenizer(batch_sentences, batch_of_second_sentences, truncation='longest_first', padding='max_length', max_length=512)
model_inputs = tokenizer(batch_sentences, batch_of_second_sentences)



print(model_inputs)

decoded = tokenizer.decode(model_inputs["input_ids"][0])

print(decoded)


train_dataset = WikiDataset(model_inputs, labels)

print("train_dataset", train_dataset.encodings, train_dataset.labels)



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




