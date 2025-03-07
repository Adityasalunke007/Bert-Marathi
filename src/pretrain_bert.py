#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load Marathi dataset
dataset = load_dataset("text", data_files="data/qa_dataset.json")

# Define model configuration
config = BertConfig(vocab_size=30522)
model = BertForMaskedLM(config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_marathi",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("./bert_marathi")

