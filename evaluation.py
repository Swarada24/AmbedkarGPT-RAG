import json
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from utils import load_corpus, chunk_documents
from metrics import *
import subprocess
import os

EMBEDDINGS = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

def build_vector_db(size):
    corpus = load_corpus()
    chunks = chunk_documents(corpus,size)

    db_path = f"db_{size}"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    docs = [
    Document(page_content=c["text"], metadata={"source": c["source"]})
    for c in chunks
]



    Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDINGS,
        persist_directory=db_path
    )

def query_llm(question, context):
    prompt = f"Answer using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = subprocess.run(
        ["ollama","run", "mistral"],
        input= prompt.encode(),
        stdout = subprocess.PIPE
    )
    return result.stdout.decode()
def evaluate(size):
    print(f"Running evaluation for {size} chunks...")

    db = Chroma(
        persist_directory = f"db_{size}",
        embedding_function=EMBEDDINGS
    )

    with open("test_dataset.json", "r", encoding = "utf-8") as f:
        data = json.load(f)["test_questions"]
    
    results = []

    for item in data:
        question = item["question"]
        truth = item["ground_truth"]
        answerable = item["answerable"]

        docs = db.similarity_search(question, k=3)
        combined_context = " ".join([d.page_content for d in docs])
        pred = query_llm(question, combined_context).strip()
        
        metrics = {
            "question_id": item["id"],
            "prediction" : pred,
            "rougeL": compute_rouge_l(pred,truth),
            "cosine_similarity" : compute_cosine(pred, truth),
            "bleu": compute_bleu(pred, truth),
            "faithfulness": compute_faithfulness(pred, combined_context),
            "relevance" : compute_relevance(pred, truth),
            "retrieved_sources" : list({d.metadata['source'] for d in docs}),
            "expected_sources" : item["source_documents"]
        }
        results.append(metrics)
    with open(f"results/results_{size}.json", "w", encoding = "utf-8") as f:
        json.dump(results,f, indent=4)
    print(f"Completed {size} evaluation")

if __name__ == "__main__":
    size = "large"
    # if DB already exists, skip rebuild
    evaluate(size)
    print("Evaluation Complete!")