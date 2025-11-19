from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_corpus, chunk_documents
from metrics import compute_faithfulness
from subprocess import run, PIPE

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory= "db_medium", embedding_function = EMBEDDINGS )
print("=CLI=")
while True:
    question = input("\nEnter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    k = 5
    docs = db.similarity_search(question, k=k)
    context = " ".join([d.page_content for d in docs])

    prompt = f"Answer using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = run(["ollama", "run", "mistral"], input = prompt.encode(), stdout= PIPE)
    answer = result.stdout.decode().strip()

    print("\n--ANS--")
    print(answer)
    print("\n--Retrieved sources--")
    print([d.metadata['source'] for d in docs])
    print("\n--Faithfulness--")
    print(compute_faithfulness(answer, context))
