import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    const validFileTypes = [".c", ".cpp", ".py", ".js", ".txt"];

    if (uploadedFile) {
      const fileExtension = uploadedFile.name.split(".").pop().toLowerCase();
      if (validFileTypes.includes(`.${fileExtension}`)) {
        setFile(uploadedFile);
        setErrorMessage("");
        setResult("");
      } else {
        setFile(null);
        setErrorMessage(
          "Invalid file type. Please upload a .c, .cpp, .py, .js, or .txt file."
        );
        setResult("");
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://localhost:8000/uploadfile/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const data = res.data;

      if (data && data.prediction) {
        setResult(`Analysis result: ${data.prediction}`);
      } else {
        setResult(data.message || `File uploaded: ${data.filename}`);
      }
    } catch (err) {
      console.error("Upload error:", err);
      setResult("Error uploading file. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div
      style={{
        padding: "2rem",
        fontFamily: "'Roboto', sans-serif",
        maxWidth: "800px",
        margin: "auto",
        textAlign: "center",
        backgroundColor: "#f9f9f9",
        borderRadius: "10px",
        boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
      }}
    >
      <h1 style={{ color: "#333", marginBottom: "1rem" }}>Code Safety Checker</h1>
      <p style={{ color: "#555", marginBottom: "2rem" }}>
        Upload your code file to check for vulnerabilities.
      </p>

      <input
        type="file"
        accept=".c,.cpp,.py,.js,.txt"
        onChange={handleFileChange}
        style={{
          padding: "0.5rem",
          border: "1px solid #ccc",
          borderRadius: "5px",
          marginBottom: "1rem",
        }}
      />
      <br />

      {errorMessage && (
        <p style={{ color: "red", marginBottom: "1rem" }}>{errorMessage}</p>
      )}

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          padding: "0.75rem 1.5rem",
          backgroundColor: loading ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: loading ? "not-allowed" : "pointer",
          fontSize: "1rem",
        }}
      >
        {loading ? "Checking..." : "Check File"}
      </button>

      <br />
      <br />
      {result && (
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem",
            backgroundColor: result.includes("VULNERABLE") ? "#f8d7da" : "#e9f7ef",
            border: result.includes("VULNERABLE") ? "1px solid #f5c6cb" : "1px solid #d4edda",
            borderRadius: "5px",
            color: result.includes("VULNERABLE") ? "#721c24" : "#155724",
          }}
        >
          <strong>Result:</strong> {result}
        </div>
      )}
    </div>
  );
}

export default App;
