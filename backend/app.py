from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Define where you want to save the file
        save_path = f"uploaded_files/{file.filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Write the file to the specified location
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Return the filename and a success message
        return JSONResponse(content={"filename": file.filename, "message": "File uploaded and saved successfully!"})
    except Exception as e:
        return JSONResponse(content={"message": "File upload failed", "error": str(e)}, status_code=500)
