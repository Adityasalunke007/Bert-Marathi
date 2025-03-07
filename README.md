## Bert-Marathi

This repository contains code for training a BERT model from scratch on Marathi language data, fine-tuning it for question-answering (QA), and deploying it as a cloud-native solution on Azure.

### Project Structure:
```
Bert-Marathi/
├── README.md               # Documentation
├── requirements.txt        # Dependencies
├── data/
│   ├── qa_dataset.json     # Sample QA dataset
├── src/
│   ├── pretrain_bert.py    # BERT Pretraining script
│   ├── fine_tune_qa.py     # Fine-tuning for QA
│   ├── inference.py        # Model inference script
│   ├── deployment/
│   │   ├── azure_deploy.yaml  # Azure deployment config
```

### Setup & Installation
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Training & Fine-Tuning
```bash
# Pretrain the BERT model on Marathi
python src/pretrain_bert.py

# Fine-tune for Question-Answering
python src/fine_tune_qa.py
```

### Running Inference
```bash
python src/inference.py --question "मराठी कोणत्या राज्याची प्रमुख भाषा आहे?"
```

### Deploying on Azure
```bash
az deployment group create --resource-group <resource-group> --template-file src/deployment/azure_deploy.yaml
```
