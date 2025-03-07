#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("json", data_files="data/qa_dataset.json")

# Load pretrained model
model = BertForQuestionAnswering.from_pretrained("./bert_marathi")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_marathi_qa",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("./bert_marathi_qa")

