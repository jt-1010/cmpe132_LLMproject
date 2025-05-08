from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, RobertaTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
# Load model and tokenizer
model_path = "C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/fine_tuned_codet5"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load and preprocess test data
dataset = load_dataset("csv", data_files="C:/Users/Jeremy/Desktop/cmpe132_LLMproject/training/dataset/formAI_dataset.csv")["train"]

# Balance and split
vul = dataset.filter(lambda x: x["Vulnerability type"] == "VULNERABLE")
non_vul = dataset.filter(lambda x: x["Vulnerability type"] != "VULNERABLE")
min_size = min(len(vul), len(non_vul))
vul = vul.shuffle(seed=42).select(range(min_size))
non_vul = non_vul.shuffle(seed=42).select(range(min_size))
balanced = concatenate_datasets([vul, non_vul]).shuffle(seed=42)
split = balanced.train_test_split(test_size=0.2, seed=42)
dev_test = split["test"].train_test_split(test_size=0.5, seed=42)
test_data = dev_test["test"]

# Preprocessing
def preprocess_function(examples):
    inputs = [f"Classify: {code if code else 'N/A'}" for code in examples["Source code"]]
    targets = ["vulnerable" if t == "VULNERABLE" else "not vulnerable" for t in examples["Vulnerability type"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=10)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize
tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

# Metrics function
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred_bin = [1 if p.strip().lower() == "vulnerable" else 0 for p in decoded_preds]
    label_bin = [1 if l.strip().lower() == "vulnerable" else 0 for l in decoded_labels]
    acc = accuracy_score(label_bin, pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(label_bin, pred_bin, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Define dummy training args for evaluation
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,
    dataloader_drop_last=False,
    fp16=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
)

# Evaluate
print("Evaluating on test set...")
results = trainer.evaluate(eval_dataset=tokenized_test)
print("Test Set Results:", results)
