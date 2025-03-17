# PDF Summarizer and Query Answering Bot using RAG & LangChain

This project implements Retrieval-Augmented Generation (RAG) to summarize PDFs and answer user queries based on their content. The solution is integrated with a Telegram bot for easy interaction. It uses LangChain, HuggingFace Embeddings, and Chroma to handle document processing and querying without relying on OpenAI or other external LLM APIs.

## Features

- **PDF Loading**: Extracts text from PDFs dynamically using PyPDFLoader.
- **Automatic Text Splitting**: Handles large PDFs by splitting content into chunks with RecursiveCharacterTextSplitter.
- **Embeddings**: Generates vector embeddings using HuggingFace’s `all-mpnet-base-v2` model.
- **Vector Store**: Stores and searches document chunks using Chroma.
- **Context-aware Query Answering**: Answers questions based on document context using a custom LLM pipeline.
- **Telegram Bot**: Directly chat with the bot to upload PDFs and ask questions without dealing with file paths manually.

## Tech Stack

- Python
- LangChain
- Hugging Face Transformers
- ChromaDB
- PyPDFLoader
- Telegram Bot API

## Project Structure

```
├── chroma_db/                   # Local directory for vector store
├── documents/                   # Folder where uploaded PDFs are stored temporarily
├── bot.py                       # Telegram bot integration script
├── main.py                      # Core document processing and RAG logic
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/palaklohade/pdf_summarizer.git
cd pdf_summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and Initialize Embedding Model in Code

```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
```

## Usage

### 1. Telegram Bot Interaction

- Upload the PDF directly to the Telegram bot.
- Ask questions like "What is this document about?" or any other context-specific query.
- The bot processes the document, stores embeddings, and responds based on document context.

### 2. Manual Usage (Without Bot) - For Testing

**Loading and Splitting PDFs**

```python
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(path):
    loader = PyPDFLoader(path)
    return loader.load()

documents = load_document("documents/sample.pdf")
chunks = split_documents(documents)  # Assuming split_documents() is defined
```

**Creating Vector Store**

```python
from langchain.vectorstores import Chroma

def add_to_chroma(chunks):
    db = Chroma(persist_directory="./chroma_db", embedding_function=get_embedding_function())
    db.add_documents(chunks, ids=[str(i) for i in range(len(chunks))])
    db.persist()
    return db

db = add_to_chroma(chunks)
```

**Querying**

```python
query_text = "What is this document about?"
results = db.similarity_search_with_score(query_text, k=3)
context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

prompt_template = """
Answer the question based only on the following context:
{context}
___
Answer the question based on the above context: {question}
"""

from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(prompt_template)
prompt = prompt_template.format_prompt(context=context_text, question=query_text)
```

## Example Output

**Query:** "What is this document about?"  
**Answer:** "The document is about recent research in NLP and its real-world applications."

## Future Enhancements

- Add support for multiple PDFs and bulk uploads.
- Fine-tune LLM for domain-specific responses.
- Add a web interface for a broader user experience beyond Telegram.

---

If you have any feedback or ideas to improve this, feel free to open an issue or contribute!

---
