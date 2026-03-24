# Project Explanation: DocBase Chatbot

## Project Overview

This project is a multi-modal conversational AI application that allows users to chat with both unstructured documents (like PDFs and text files) and structured SQLite databases. The project provides a unified, user-friendly web interface built with Streamlit, where users can upload their data sources and ask questions in natural language. The project is a practical demonstration of two powerful large language model techniques: Retrieval-Augmented Generation (RAG) for documents and Text-to-SQL for databases.

## Key Features

*   **Dual-Capability Chatbot:** The chatbot can seamlessly switch between interacting with documents and databases, providing a single interface for different data querying needs.
*   **Retrieval-Augmented Generation (RAG):** For documents, the project implements a full RAG pipeline. It ingests documents, splits them into chunks, creates vector embeddings, and stores them in a FAISS vector store. When a user asks a question, the system retrieves the most relevant document chunks and uses them as context for an LLM to generate a well-grounded answer.
*   **Text-to-SQL:** For databases, the chatbot can convert a user's natural language question into a SQL query. It does this by first extracting the database schema and then feeding the schema and the user's question to an LLM.
*   **Interactive Web Interface:** A Streamlit application provides an easy-to-use interface for users to upload their documents or databases and start a conversation.
*   **Modular and Standalone Scripts:** The core logic for document and database interaction is also available as standalone command-line scripts (`talk_to_documents.py` and `talk_to_databases.py`), which is useful for testing and demonstration.
*   **Local LLM Integration:** The project is designed to run with locally hosted large language models via Ollama, making it a private and self-contained solution.

## Technologies Used

*   **AI Framework:** LangChain for orchestrating the RAG and Text-to-SQL pipelines.
*   **Web Framework:** Streamlit for the interactive web application.
*   **Large Language Models:** Locally hosted models via Ollama (e.g., Llama 3.1, Llama 3.2).
*   **Vector Database:** FAISS for efficient similarity search of document embeddings.
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` for creating vector embeddings from text.
*   **Programming Languages:** Python.

## Directory Structure

*   `app.py`: The main entry point for the Streamlit web application. It integrates both the document and database chat functionalities.
*   `talk_to_documents.py`: A command-line script for interacting with documents. It reads documents from the `documents/` directory and can create a persistent vector store.
*   `talk_to_databases.py`: A standalone script that demonstrates the full text-to-SQL functionality, including query generation and execution.
*   `documents/`: A directory where documents can be placed for the `talk_to_documents.py` script.
*   `vectorstore/`: The directory where the persistent FAISS vector store is saved.
*   `test_ingest.py` & `test_llm.py`: Utility scripts for testing different configurations.
*   `requirements.txt`: A list of Python dependencies.

## How it Works

**Document Chat (RAG):**
1.  **Ingestion:** When a user uploads documents, they are loaded and split into smaller chunks.
2.  **Vectorization:** Each chunk is converted into a numerical representation (a vector embedding) using the Sentence-Transformers model.
3.  **Indexing:** The embeddings are stored in a FAISS vector store for fast retrieval.
4.  **Retrieval & Generation:** When a user asks a question, the system first retrieves the most relevant document chunks from the vector store. Then, it passes these chunks, along with the user's question, to the LLM, which generates a final answer based on the provided context.

**Database Chat (Text-to-SQL):**
1.  **Schema Extraction:** The application connects to the user's SQLite database and extracts its schema (table names, column names, etc.).
2.  **Query Generation:** The schema and the user's natural language question are formatted into a prompt and sent to the LLM. The LLM then generates a SQL query that should answer the user's question.
3.  **Display Query:** The generated SQL query is displayed to the user. In the main `app.py`, the query is not executed for safety, but the `talk_to_databases.py` script demonstrates how it can be executed.

## Key Techniques

*   **Retrieval-Augmented Generation (RAG):** This is a powerful technique that grounds the LLM's responses in a specific set of documents, reducing hallucinations and allowing the chatbot to answer questions about information it was not originally trained on.
*   **Text-to-SQL:** This is the process of converting natural language into SQL queries. It's a complex task that requires the LLM to understand both the user's intent and the structure of the database.
*   **Vector Embeddings and Similarity Search:** The project uses vector embeddings to represent the meaning of text and a FAISS vector store to perform efficient similarity searches, which is the core of the RAG retrieval step.
*   **Prompt Engineering:** The project uses carefully crafted prompts to guide the LLM in both the RAG and Text-to-SQL tasks. For Text-to-SQL, providing the database schema in the prompt is a critical piece of prompt engineering.
*   **Streamlit for Rapid Prototyping:** The use of Streamlit demonstrates how to quickly build and deploy interactive web applications for AI and machine learning projects.
