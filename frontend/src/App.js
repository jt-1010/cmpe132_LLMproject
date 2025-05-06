// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    console.log(uploadedFile.name); // Log the filename to check
    const validFileTypes = ['.c', '.cpp', '.py', '.js', '.txt'];
  
    const fileExtension = uploadedFile.name.split('.').pop().toLowerCase();
    console.log(fileExtension); // Log the file extension
    
    if (validFileTypes.includes(`.${fileExtension}`)) {
      setFile(uploadedFile);
      setErrorMessage('');
    } else {
      setFile(null);
      setErrorMessage('Invalid file type. Please upload a .c, .cpp, .py, .js, or .txt file.');
    }
    setResult('');
  };
  

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    console.log("Uploading file:", file.name); // Log the file name
  
    try {
      const res = await axios.post('http://localhost:8000/uploadfile/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log("Response:", res); // Log the response from the backend
      setResult(res.data.message || `File uploaded successfully: ${res.data.filename}`);
    } catch (err) {
      console.error(err);
      setResult('Error uploading file');
    }
    setLoading(false);
  };
  

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial' }}>
      <h2>Code Safety Checker</h2>
      <input 
        type="file" 
        accept=".c,.cpp,.py,.js,.txt" 
        onChange={handleFileChange} 
      />
      <br /><br />
      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
      <button onClick={handleUpload} disabled={!file || loading}>
        {loading ? "Checking..." : "Check File"}
      </button>
      <br /><br />
      {result && <p><strong>Result:</strong> {result}</p>}
    </div>
  );
}

export default App;
