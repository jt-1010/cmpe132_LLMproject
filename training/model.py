from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import accuracy_score,  precision_recall_fscore_support
import numpy as np
import torch
import torch.nn as nn

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# Load dataset
dataset = load_dataset("csv", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/formAI_dataset.csv")["train"]

# Filter vulnerable and not vulnerable examples
vulnerable_data = dataset.filter(lambda x: x["Vulnerability type"] == "VULNERABLE")
not_vulnerable_data = dataset.filter(lambda x: x["Vulnerability type"] != "VULNERABLE")
min_size = min(len(vulnerable_data), len(not_vulnerable_data))

vulnerable_data = vulnerable_data.shuffle(seed=42).select(range(min_size))
not_vulnerable_data = not_vulnerable_data.shuffle(seed=42).select(range(min_size))
balanced_dataset = concatenate_datasets([vulnerable_data, not_vulnerable_data])
balanced_dataset = balanced_dataset.shuffle(seed=42)
# Split dataset
split_dataset = balanced_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
temp_data = split_dataset["test"]

dev_test_split = temp_data.train_test_split(test_size=0.5, seed=42)
dev_data = dev_test_split["train"]
test_data = dev_test_split["test"]

dataset = DatasetDict({"train": train_data, "dev": dev_data, "test": test_data})

# Use CodeBERT
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = []
    for source_code, error_type, vulnerability_type in zip(
        examples["Source code"], examples["Error type"], examples["Vulnerability type"]
    ):
        if source_code is None:
            source_code = "N/A"

        input_text = "Source code: " + source_code

        inputs.append(input_text)

    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["labels"] = [1 if vt == "VULNERABLE" else 0 for vt in examples["Vulnerability type"]]
    return tokenized_inputs

# Tokenize
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Count label distribution to compute class weights
labels = tokenized_dataset["train"]["labels"]
num_positives = sum(labels)
num_negatives = len(labels) - num_positives
weight_for_0 = num_positives / (num_positives + num_negatives)
weight_for_1 = num_negatives / (num_positives + num_negatives)
class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)


# Load model with modified class
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy='epoch',
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=5e-5, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    save_total_limit=1,  # optional: keep only best model
    logging_steps=50,
    weight_decay=0.01,
    eval_steps=500,
    warmup_steps=500,
    fp16=True
)

# Custom trainer with weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <-- Accept **kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    
# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# Train
trainer.train()

# Save model
trainer.save_model("./fine_tuned_codebert_new")
tokenizer.save_pretrained("./fine_tuned_codebert_new")

# Evaluate
print("Evaluating on training set...")
train_results = trainer.evaluate(eval_dataset=tokenized_dataset["train"])
print("Training Set Results:", train_results)

print("Evaluating on development set...")
dev_results = trainer.evaluate(eval_dataset=tokenized_dataset["dev"])
print("Development Set Results:", dev_results)
