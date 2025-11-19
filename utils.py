import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_corpus(corpus_path= "corpus"):
    docs = {}
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs

def get_text_splitter(size="small"):
    if size =="small":
        chunk_size = 250
    elif size == "medium":
        chunk_size = 550
    elif size == "large":
        chunk_size = 900
    else:
        raise ValueError("Invalid size")
    
    return RecursiveCharacterTextSplitter( 
        chunk_size=chunk_size,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", ".", " ", "" ] )

def chunk_documents(docs, size):
    splitter = get_text_splitter(size)
    chunks = []
    for filename, text in docs.items():
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "source": filename})
    return chunks

