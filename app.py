import streamlit as st
import os
import sqlite3
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'db_faiss' not in st.session_state:
    st.session_state.db_faiss = None
if 'schema' not in st.session_state:
    st.session_state.schema = None

# Function to extract schema from the database
def extract_schema(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_info = []
        for table_name in tables:
            cursor.execute(f"PRAGMA table_info({table_name[0]})")
            columns = cursor.fetchall()
            schema_info.append(f"Table: {table_name[0]}")
            for column in columns:
                schema_info.append(f"  - {column[1]} ({column[2]})")
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        st.error(f"Error extracting schema: {str(e)}")
        return None

# Function to generate SQL query
def generate_sql_query(schema, question):
    llm = Ollama(model="llama3.1:8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that generates SQL queries based on user requests. You have access to the following database schema:\n{schema}"),
        ("human", "Based on this schema, generate a SQL query to answer the following question: {question}"),
        ("ai", "Here's the SQL query to answer your question:\n")
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.run(schema=schema, question=question)
        return result.strip()
    except Exception as e:
        st.error(f"Error generating SQL query: {str(e)}")
        return None

# Function to process documents and create vector store
def process_documents(uploaded_files):
    all_documents = []
    for file in uploaded_files:
        file_path = os.path.join("temp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue
        
        all_documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_faiss = FAISS.from_documents(split_documents, embeddings)
    return db_faiss

# Streamlit app
def main():
    st.title("DocBase Chatbot")

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True, type=['pdf', 'txt'])
        db_file = st.file_uploader("Upload SQLite database file", type=['db'])
        
        if st.button("Process Files"):
            if uploaded_files:
                st.session_state.db_faiss = process_documents(uploaded_files)
                st.success("Documents processed successfully!")
            if db_file:
                st.session_state.schema = extract_schema(db_file.name)
                st.success("Database schema extracted successfully!")

    # Main chat interface
    st.header("Chat with Your Documents and Database")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents or database"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.db_faiss is None:
            st.error("Please upload and process documents before asking questions.")
            return

        # Create retrieval chain
        llm = Ollama(model="llama3.2:7b")
        chat_prompt = ChatPromptTemplate.from_template('''
        Answer the following question based on the provided documents and chat history.
        Think step by step before providing a detailed answer.
        I will tip you $1000 if the user finds the answer helpful.
        <context>
        {context}
        </context>
        <chat_history>
        {chat_history}
        </chat_history>
        Question: {input}
        ''')
        doc_chain = create_stuff_documents_chain(llm, chat_prompt)
        retriever = st.session_state.db_faiss.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Document Q&A
            doc_response = retrieval_chain.invoke({
                "input": prompt,
                "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            })
            full_response += f"Document Answer: {doc_response['answer']}\n\n"
            response_placeholder.markdown(full_response)

            # Database Query (if schema is available)
            if st.session_state.schema:
                sql_query = generate_sql_query(st.session_state.schema, prompt)
                if sql_query:
                    full_response += f"Generated SQL Query: {sql_query}\n\n"
                    response_placeholder.markdown(full_response)
                    # Note: We're not executing the SQL query here for privacy reasons
                    full_response += "Note: For privacy reasons, we don't execute the SQL query on actual data.\n"
                    response_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()