import os
import tempfile
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
The context will be passed as "Context:", user question will be passed as "Question:"
The answers will response solely on the information provided in the context. Any external knowledge or assumptions 
will not present answers.
"""
# Processes multiple uploaded PDF files by converting them to text chunks.
def process_document(uploaded_files: list[UploadedFile]) -> list[Document]:
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        all_docs.extend(text_splitter.split_documents(docs))
    
    return all_docs

#Gets or creates a ChromaDB collection for vector storage.
def get_vector_collection() -> chromadb.Collection:    
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

## Adds multiple document splits to a vector collection.
def add_to_vector_collection(all_splits: list[Document]):    
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"doc_{idx}")
    
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("All documents added to the vector store!")

# Queries the vector collection with a given prompt to retrieve relevant documents.
def query_collection(prompt: str, n_results: int = 10):
    
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):   
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break


## Re-ranks documents using a cross-encoder model for more accurate relevance scoring.
def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:    
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    
    return relevant_text, relevant_text_ids


# main func.
if __name__ == "__main__":    
    with st.sidebar:
        st.set_page_config(page_title="Derek's Offline Knowledge Base - RAG")
        uploaded_files = st.file_uploader(
            "📑 Upload PDF files for Q&A", type=["pdf"], accept_multiple_files=True
        )
        process = st.button("Upload")
        
        if uploaded_files and process:
            st.balloons()
            all_splits = process_document(uploaded_files)
            add_to_vector_collection(all_splits)
    
    # Question and Answer Area
    st.subheader("_Derek's Local Knowledge Base - Offline AI+RAG_")
    prompt = st.text_area("**Ask a question related to the uploaded documents:**")
    ask = st.button("Submit")
    
    if ask and prompt:
        st.balloons()
        results = query_collection(prompt)
        context = results.get("documents", [""])[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)
        
        with st.expander("See retrieved documents"):
            st.write(results)
        
        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
