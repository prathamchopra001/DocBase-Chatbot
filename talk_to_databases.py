import sqlite3
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model="llama3.1:8b")

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
        print(f"Error extracting schema: {str(e)}")
        return None

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that generates SQL queries based on user requests. You have access to the following database schema:\n{schema}"),
    ("human", "Based on this schema, generate a SQL query to answer the following question: {question}"),
    ("ai", "Here's the SQL query to answer your question:\n")
])

chain = LLMChain(llm=llm, prompt=prompt)

def generate_sql_query(schema, question):
    try:
        result = chain.run(schema=schema, question=question)
        return result.strip()
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        return None

def execute_sql_query(db_file, query):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
        return None

db_file = 'documents/new.db'
schema = extract_schema(db_file)

user_question = "Find me the registration id of the Hackathon"
sql_query = generate_sql_query(schema, user_question)
print(f"Generated SQL Query: {sql_query}")

if sql_query:
    results = execute_sql_query(db_file, sql_query)
    print(f"Query Results: {results}")