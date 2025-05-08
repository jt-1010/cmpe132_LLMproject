from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# Load dataset
dataset = load_dataset("csv", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/formAI_dataset.csv")["train"]

# Balance dataset
vulnerable_data = dataset.filter(lambda x: x["Vulnerability type"] == "VULNERABLE")
not_vulnerable_data = dataset.filter(lambda x: x["Vulnerability type"] != "VULNERABLE")
min_size = min(len(vulnerable_data), len(not_vulnerable_data))
vulnerable_data = vulnerable_data.shuffle(seed=42).select(range(min_size))
not_vulnerable_data = not_vulnerable_data.shuffle(seed=42).select(range(min_size))
balanced_dataset = concatenate_datasets([vulnerable_data, not_vulnerable_data]).shuffle(seed=42)

# Split dataset
split_dataset = balanced_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
temp_data = split_dataset["test"]
dev_test_split = temp_data.train_test_split(test_size=0.5, seed=42)
dev_data = dev_test_split["train"]
test_data = dev_test_split["test"]
dataset = DatasetDict({"train": train_data, "dev": dev_data, "test": test_data})

# Load CodeT5 tokenizer and model
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

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

# Tokenize datasets
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Compute evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = [1 if p.strip().lower() == "vulnerable" else 0 for p in decoded_preds]
    true = [1 if l.strip().lower() == "vulnerable" else 0 for l in decoded_labels]

    acc = accuracy_score(true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_codet5",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=200,
    weight_decay=0.01,  # L2 regularization
    logging_dir="./logs_codet5",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    learning_rate=5e-5,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    load_best_model_at_end=True,  # Load the best model at the end
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)  # Stop training early if no improvement after 2 evaluations

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],  # Add early stopping
)

# Train the model
trainer.train()

# Save model and tokenizer
trainer.save_model("./fine_tuned_codet5_real")
tokenizer.save_pretrained("./fine_tuned_codet5_real")

# Evaluate
print("Evaluating on training set...")
train_results = trainer.evaluate(eval_dataset=tokenized_dataset["train"])
print("Training Set Results:", train_results)

print("Evaluating on development set...")
dev_results = trainer.evaluate(eval_dataset=tokenized_dataset["dev"])
print("Development Set Results:", dev_results)
