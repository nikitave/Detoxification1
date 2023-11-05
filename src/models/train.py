import pandas as pd
import torch
from data.dataset import ToxicityDataset
from torch.utils.data import random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments


data = pd.read_csv("./data/filtered.tsv", delimiter="\t")
test_data = data.sample(frac=0.2, random_state=42)
data = data.drop(test_data.index)
data = data.drop("Unnamed: 0", axis=1)

test_data.to_csv("./data/test_dataset.tsv", index=False, sep='\t')

tokenizer = T5Tokenizer.from_pretrained("t5-base")
dataset = ToxicityDataset(data, tokenizer, max_length=120, task="generation")
train_dataset, test_dataset, validate_dataset = random_split(dataset, [.7, .2, .1])
torch.save(test_dataset, f="test_dataset.obj")

model_name = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
training_args = TrainingArguments(
    output_dir="./t5_toxicity_finetuned",
    per_device_train_batch_size=54,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=15000,
    save_steps=15000,
    logging_steps=15000
)

def model_init():
    return T5ForConditionalGeneration.from_pretrained(model_name)

from transformers import Trainer, TrainingArguments

# Define the trainer and training arguments
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset
)

# Train the model
trainer.train()

model_dir = "./models"
trainer.save_model(model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained('./models')