from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch

print("Using GPU:", torch.cuda.get_device_name(0))

# Load model and tokenizer
## Default Model
# model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
## Model Trained Once
# model = AutoModelForSequenceClassification.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert_new")
# tokenizer = AutoTokenizer.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert_new")
## Model Trained Twice
model = AutoModelForSequenceClassification.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert_better")
tokenizer = AutoTokenizer.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert_better")
# Load and shuffle the original dataset
# dataset = load_dataset("csv", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/formAI_dataset.csv")["train"]
dataset = load_dataset("json", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/diversevul_20230702.json")["train"]
dataset = dataset.shuffle(seed=42)

# Use 10% as test datavfor DiverseVul
test_data = dataset.select(range(int(0.1 * len(dataset))))
def preprocess_function(examples):
    inputs = []
    targets = []
    for func, target in zip(examples["func"], examples["target"]):
        func = func if func else "N/A"
        inputs.append(f"Classify: {func}")
        targets.append("vulnerable" if target == 1 else "not vulnerable")

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    
    label_map = {"not vulnerable": 0, "vulnerable": 1}
    model_inputs["labels"] = [label_map[label] for label in targets]

    return model_inputs
print(dataset.column_names)

# Preprocessing function for FormAI
# def preprocess_function(examples):
#     inputs = []
#     targets = []
#     for source_code, vulnerability_type in zip(examples["Source code"], examples["Vulnerability type"]):
#         source_code = source_code if source_code else "N/A"
#         inputs.append(f"Classify: {source_code}")
#         targets.append("vulnerable" if vulnerability_type == "VULNERABLE" else "not vulnerable")

#     model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    
#     ### Testing on the CodeT5
#     # with tokenizer.as_target_tokenizer():
#     #     labels = tokenizer(targets, padding="max_length", truncation=True, max_length=10)
#     # model_inputs["labels"] = labels["input_ids"]
#     ###
    
#     ### Testing on the CodeBERT
#     label_map = {"not vulnerable": 0, "vulnerable": 1}
#     model_inputs["labels"] = [label_map[label] for label in targets]
#     ###
    
#     return model_inputs
# Tokenize the test data
tokenized_test = test_data.map(preprocess_function, batched=True)

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model (now the model doesn’t see labels until after prediction)
results = trainer.evaluate(eval_dataset=tokenized_test)
print("Evaluation Results on Test Subset:", results)
