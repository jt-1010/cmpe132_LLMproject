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

      // Ensure the response is in JSON format
      const data = res.data;

      // Check if the response contains the expected result
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
        fontFamily: "Arial",
        maxWidth: "600px",
        margin: "auto",
      }}
    >
      <h2>Code Safety Checker</h2>

      <input
        type="file"
        accept=".c,.cpp,.py,.js,.txt"
        onChange={handleFileChange}
      />
      <br />
      <br />

      {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          padding: "0.5rem 1rem",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          cursor: "pointer",
        }}
      >
        {loading ? "Checking..." : "Check File"}
      </button>

      <br />
      <br />
      {result && (
        <p>
          <strong>Result:</strong> {result}
        </p>
      )}
    </div>
  );
}

export default App;
