import torch
from torch.utils.data import Dataset


class ToxicityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, task="classification"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["reference"]
        target = self.data.iloc[index]["translation"]
        label = self.data.iloc[index]["ref_tox"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        if self.task == "classification":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        elif self.task == "generation":
            target_encoding = self.tokenizer(
                target,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": target_encoding["input_ids"].squeeze(),
            }
        
