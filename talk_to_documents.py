import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


DB_FAISS_PATH = "vectorstore"
docs_path = 'documents'
chat_history = []


files = os.listdir(docs_path)


pdf_documents = []
txt_documents = []


if len(files) == 1:
    if files[0].endswith('.pdf'):
        pdf_loader = PyPDFLoader(os.path.join(docs_path, files[0]))
        pdf_documents = pdf_loader.load()
    
    elif files[0].endswith('.txt'):
        txt_loader = TextLoader(os.path.join(docs_path, files[0]))
        txt_documents = txt_loader.load()
    
    else:
        print("Upload only pdf or txt files")
        exit()


elif len(files) > 1:
    pdf_loader = DirectoryLoader(docs_path, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    txt_loader = DirectoryLoader(docs_path, glob='*.txt', loader_cls=TextLoader)
    txt_documents = txt_loader.load()


# Combine pdf and txt documents
all_documents = pdf_documents + txt_documents


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(all_documents)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_faiss = FAISS.from_documents(split_documents, embeddings)
db_faiss.save_local(DB_FAISS_PATH)


llm = Ollama(model="llama3.1:8b")


prompt = ChatPromptTemplate.from_template('''
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


doc_chain = create_stuff_documents_chain(llm, prompt)


retriever = db_faiss.as_retriever()


retrieval_chain = create_retrieval_chain(retriever, doc_chain)


while True:
    question = input("Enter your question: ")
    chat_history.append(f"User: {question}")
    
    response = retrieval_chain.invoke({"input": question, "chat_history": "\n".join(chat_history)})
    answer = response["answer"]
    print(f"---------------------------------------------------------------------\nAnswer: {answer}")
    
    chat_history.append(f"Assistant: {answer}")