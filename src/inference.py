#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import sys

# Load model and tokenizer
model = BertForQuestionAnswering.from_pretrained("./bert_marathi_qa")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )

if __name__ == "__main__":
    question = sys.argv[1]
    context = "मराठी ही भारताच्या महाराष्ट्र राज्यातील प्रमुख भाषा आहे."
    print(answer_question(question, context))

