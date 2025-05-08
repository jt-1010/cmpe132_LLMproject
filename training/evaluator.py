from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score
import numpy as np
import torch

print("Using GPU:", torch.cuda.get_device_name(0))

# Load fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert")
tokenizer = AutoTokenizer.from_pretrained("C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codebert")

# Load and shuffle the original dataset
dataset = load_dataset("csv", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/formAI_dataset.csv")["train"]
dataset = dataset.shuffle(seed=42)

# Use 10% as test data
test_data = dataset.select(range(int(0.1 * len(dataset))))

# Preprocessing function
def preprocess_function(examples):
    inputs = []
    targets = []
    for source_code, vulnerability_type in zip(examples["Source code"], examples["Vulnerability type"]):
        source_code = source_code if source_code else "N/A"
        inputs.append(f"Classify: {source_code}")
        targets.append("vulnerable" if vulnerability_type == "VULNERABLE" else "not vulnerable")

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=10)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# Tokenize the test data
tokenized_test = test_data.map(preprocess_function, batched=True)

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model (now the model doesn’t see labels until after prediction)
results = trainer.evaluate(eval_dataset=tokenized_test)
print("Evaluation Results on Test Subset:", results)
