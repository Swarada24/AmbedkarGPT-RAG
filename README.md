# 📘 AmbedkarGPT – RAG Evaluation Framework (Assignment 2)

This repository contains **Assignment 2** of the AmbedkarGPT Intern Task.

The goal is to build a **complete evaluation system** for a Retrieval-Augmented Generation (RAG) pipeline using:

- **LangChain**
- **ChromaDB** (local vector database)
- **HuggingFace Embeddings** (sentence-transformers/all-MiniLM-L6-v2)
- **Ollama + Mistral 7B** (offline LLM)
- **Evaluation Metrics** (ROUGE, BLEU, Cosine Similarity, Hit Rate, MRR, etc.)

The system automatically evaluates the RAG pipeline using **25 predefined questions** and compares **three chunking strategies**.

---

# 🧠 What This Project Does

This project automatically:

1. Loads the **6 Ambedkar speech documents**.
2. Splits them into **three chunk sizes**:
   - Small (200–300 characters)
   - Medium (500–600 characters)
   - Large (800–1000 characters)
3. Builds **three Chroma vector databases**.
4. Runs RAG for **25 test questions**.
5. Generates answers using **Mistral 7B (via Ollama)**.
6. Computes:
   - Hit Rate
   - Mean Reciprocal Rank (MRR)
   - Precision@K
   - ROUGE-L
   - BLEU
   - Cosine Similarity
   - Faithfulness
   - Relevance
7. Saves results under `/results/`
8. Generates a combined report
9. (You write the final analysis in results_analysis.md)

---

# 📁 Project Structure

AmbedkarGPT-Intern-Task/
│
├── corpus/
│ ├── speech1.txt
│ ├── speech2.txt
│ ├── speech3.txt
│ ├── speech4.txt
│ ├── speech5.txt
│ └── speech6.txt
│
├── test_dataset.json
│
├── evaluation.py
├── utils.py
├── metrics.py
│
├── db_small/
├── db_medium/
├── db_large/
│
├── results/
│ ├── results_small.json
│ ├── results_medium.json
│ ├── results_large.json
│ └── combined_report.json
│
├── results_analysis.md
├── requirements.txt
└── README.md


---

# 🔧 Installation

## 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git

cd AmbedkarGPT-Intern-Task


## 2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate # Windows

or

source venv/bin/activate # Mac/Linux

## 3️⃣ Install all dependencies
pip install -r requirements.txt


---

# 🤖 Install Ollama + Mistral 7B

### 1. Install Ollama  
Download from: https://ollama.com/download

### 2. Pull Mistral 7B

Make sure the model appears in:

---

# 🚀 Running Full Evaluation

To run evaluation for ALL chunk sizes:


This will:

- Build vector DBs (db_small, db_medium, db_large)
- Run RAG on all 25 questions
- Compute all metrics
- Save results inside `/results/`
- Generate combined_report.json

You should see:


---

# 📊 Output Files

### Located in `/results/`:
- **results_small.json**
- **results_medium.json**
- **results_large.json**
- **combined_report.json**

### You must then write:
- **results_analysis.md**

This file answers:
- Which chunking strategy works best?
- What is the system’s accuracy?
- Common failure cases?
- What improvements are recommended?

---

# 🎯 Evaluation Metrics Used

### Retrieval Metrics
- Hit@5
- Precision@5
- Mean Reciprocal Rank (MRR)

### Answer Quality
- ROUGE-L
- BLEU
- Faithfulness (context-grounded)
- Relevance

### Semantic Metrics
- Cosine Similarity (sentence-transformers)

---

# 🧪 (Optional) Run Interactive Q&A (CLI)

After DBs are built:

This loads the **medium DB** and lets you ask questions manually.

---

# 🏁 Final Deliverables

You must upload to GitHub:

- [x] evaluation.py  
- [x] utils.py  
- [x] metrics.py  
- [x] test_dataset.json  
- [x] corpus/  
- [x] results/  
- [x] results_analysis.md  
- [x] requirements.txt  
- [x] README.md  

Then submit the GitHub link.

---

# 🎉 Conclusion

This project evaluates the quality of a complete RAG pipeline using:
- Multiple chunk sizes  
- Offline vector search  
- Offline LLM inference  
- NLP metrics  
- Comparative analysis  

It provides a research-style evaluation that helps identify the **best configuration** for RAG performance.

---
