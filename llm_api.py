# llm_pipeline.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "./chroma_db"

# Load PDF dynamically
def load_document(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    print(f"File found: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

# Split document into chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# Embedding function
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return embeddings

# Add chunks to Chroma DB
def add_to_chroma(chunks: list[Document], chunk_ids: list[str]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    db.add_documents(chunks, ids=chunk_ids)
    db.persist()
    return db

# Query pipeline
def query_rag(db, query_text: str):
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    Prompt_Template = """
    {question} 
    I'll we sending the nearest three response : 
    ___
     {context}
    """

    prompt_template = ChatPromptTemplate.from_template(Prompt_Template)
    prompt = prompt_template.format_prompt(context=context_text, question=query_text)
    
    # For now return the prompt (simulate LLM response)
    return prompt.to_string()  # Replace with actual LLM call if needed

# Wrapper to process PDF
def process_pdf_with_llm(file_path):
    documents = load_document(file_path)
    chunks = split_documents(documents)
    new_chunk_ids = [str(i) for i in range(len(chunks))]
    db = add_to_chroma(chunks, new_chunk_ids)
    return db

# Answer query using generated DB
def answer_query_from_pdf(db, query_text):
    return query_rag(db, query_text)
