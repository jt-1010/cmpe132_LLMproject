from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import traceback
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
# Load model and tokenizer once on app startup
model_dir = r"C:/Users/zheng/Downloads/CMPE132-LLM/cmpe132_LLMproject/training/fine_tuned_codebert_new"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

model.eval()

def create_input(source_code, error_type="N/A", line="1"):
    try:
        line_number = int(line) - 1
        source_code_lines = source_code.splitlines()
        line_content = source_code_lines[line_number] if 0 <= line_number < len(source_code_lines) else "N/A"
    except:
        line_content = "N/A"
    
    input_text = (
        "Source code: " + source_code +
        " Line content: " + line_content +
        " Error type: " + error_type
    )
    return input_text

def predict_vulnerability(source_code):
    input_text = create_input(source_code)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "VULNERABLE" if predicted_class == 1 else "SAFE"

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file
        save_path = f"uploaded_files/{file.filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Print confirmation
        print(f"File {file.filename} saved at {save_path}")

        # Run model on the uploaded C code
        source_code = content.decode("utf-8", errors="ignore")
        result = predict_vulnerability(source_code)
        print(result)
        return JSONResponse(content={
            "filename": file.filename,
            "message": "File uploaded and processed successfully!",
            "prediction": result
        })

    except Exception as e:
        error_traceback = traceback.format_exc()
        print("Error during file upload or processing:", error_traceback)
        return JSONResponse(
            content={"message": "File upload or processing failed", "error": str(e), "traceback": error_traceback},
            status_code=500
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the Code Safety Checker API"}
