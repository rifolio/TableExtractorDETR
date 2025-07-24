# TableExtractor

<img src="https://img.shields.io/badge/-Python-blue?style=for-the-badge"><img src="https://img.shields.io/badge/-Flask-black?style=for-the-badge"><img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"><img src="https://img.shields.io/badge/-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black"><img src="https://img.shields.io/badge/-Numpy-blue?style=for-the-badge&logo=numpy&logoColor=white"><img src="https://img.shields.io/badge/-Matplotlib-darkgreen?style=for-the-badge&logo=matplotlib&logoColor=white"><img src="https://img.shields.io/badge/-Pillow-lightgrey?style=for-the-badge"><img src="https://img.shields.io/badge/-EasyOCR-orange?style=for-the-badge"><img src="https://img.shields.io/badge/-PyMuPDF-lightblue?style=for-the-badge">

A minimal, research-focused pipeline for extracting tables from PDFs using state-of-the-art DETR-based models (Table Transformer) and simple postprocessing. This project is intended as a quick testbed and reference for table extraction, not as a production-ready solution.

---

> **Want to play around with the model?**
> You can use our [Google Colab notebook](https://colab.research.google.com/drive/1ycYcKH8obluiutpk2F5EzM6mQZvg12hC?usp=sharing) to experiment interactively. The notebook includes a nice explanation and is easy to copy and use in your own Google Drive.

## 🚀 Quick Start

### 1. Run with Docker (Recommended)

**Requirements:**

- Docker installed

```bash
# Build and run the API (serves on http://localhost:5000)
docker-compose up --build
```

- PDF files placed in `pdfs/` will be accessible inside the container.
- Extracted images will be saved to `images/`.

### 2. Run with Python Virtual Environment

**Requirements:**

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/installation/)

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download models (optional, will auto-download on first run)
python local_model.py

# Start the API
python src/app/main.py
```

---

## 🖥️ Frontend

A simple static HTML frontend is provided in `frontend/index.html`.  
Open it in your browser and connect to the backend at `http://localhost:5000`.

---

## 🧠 How It Works

### Main Pipeline

1. **PDF Upload:**  
   User uploads a PDF via the frontend or API.

2. **PDF to Images:**  
   Each page is converted to an image using PyMuPDF (`src/app/pdf_reader.py`).

3. **Table Detection & Structure Recognition:**

   - **TD Model:** [Table Transformer Detection](https://huggingface.co/microsoft/table-transformer-detection)  
     Detects tables and rotated tables in the image.
   - **TSR Model:** [Table Structure Recognition](https://huggingface.co/microsoft/table-structure-recognition-v1.1-all)  
     Recognizes table structure (rows, columns, headers, spanning cells).

4. **OCR:**  
   Each detected cell is read using EasyOCR.

5. **Postprocessing:**

   - Detected bounding boxes are mapped to a grid.
   - OCR results are assembled into a 2D array (list of lists of strings).

6. **API Output:**  
   Returns the extracted table(s) as JSON.

---

## 🏷️ Model Output Classes

### Table Detection (TD) Model

- `"table"`: Standard table
- `"table rotated"`: Rotated table
- `"no object"`: No table detected

### Table Structure Recognition (TSR) Model

- `0: 'table'`
- `1: 'table column'`
- `2: 'table row'`
- `3: 'table column header'`
- `4: 'table projected row header'`
- `5: 'table spanning cell'`

---

## 📦 API Endpoints

- `POST /process`  
  Upload a PDF file (`pdf` field).  
  Returns:

  ```json
  {
    "data": [
      [
        ["cell1", "cell2", ...],
        ...
      ]
    ]
  }
  ```

- `GET /health`  
  Health check.

---

## 🗂️ Data Postprocessing Logic

- Bounding boxes for rows, columns, and spanning cells are detected.
- A grid is constructed by intersecting row and column boxes.
- Spanning cells are mapped to all grid positions they cover.
- Each cell is cropped and OCR is applied.
- The result is a rectangular 2D array, padded as needed.

> **Note:**
> For robust postprocessing, thresholding, and extracting meaning from model predictions, we strongly recommend referring to the official [Microsoft Table Transformer postprocess.py](https://github.com/microsoft/table-transformer/blob/main/src/postprocess.py). Their code covers many edge cases and implements a much more comprehensive logic for table structure extraction. As this was not the main scope of our project, we suggest using their approach for production or research-grade extraction. They also provide clear instructions for training and fine-tuning the model.

---

## ⚠️ Limitations & Recommendations

- This repo is a quick testbed, not a production system.
- For complex tables, multi-page tables, or robust extraction, refer to the above projects.
- Postprocessing is intentionally simple and may fail on edge cases.
- We suggest to use better OCR model, as easyOCR can fail in lots of casese.

---

## 🖼️ Example: Complex Tables

For examples of complex table extraction, see the [RAGFlow project](https://github.com/microsoft/table-transformer) and its documentation.

---

## 📁 Project Structure

```
TableExtractorDETR/
  ├── src/app/           # Main backend code (Flask API, inference, PDF reader)
  ├── models/            # Downloaded model weights (auto-populated)
  ├── pdfs/              # Input PDFs
  ├── images/            # Output images from PDF pages
  ├── frontend/          # Static HTML frontend
  ├── requirements.txt   # Python dependencies
  ├── Dockerfile         # Docker build
  ├── docker-compose.yml # Docker Compose config
  └── local_model.py     # Script to pre-download models
```

---

## 📚 References

- [Table Transformer (TATR) Paper](https://arxiv.org/abs/2204.08320)
- [Microsoft Table Transformer GitHub](https://github.com/microsoft/table-transformer)
- [RAGFlow](https://github.com/microsoft/table-transformer)

---

## License

MIT

---
