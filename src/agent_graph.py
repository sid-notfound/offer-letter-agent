__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
print(f"Patched sqlite3 version: {sqlite3.sqlite_version}")

import os
import csv
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from chromadb.config import Settings

# ✅ Load environment variables
load_dotenv()

# 🔑 Set up LLM
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 📄 Load employee metadata from CSV
def load_employee_data(name: str, csv_path="data/employee_metadata.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Employee Name"].strip().lower() == name.strip().lower():
                return {
                    "name": row["Employee Name"],
                    "position": row["Department"],
                    "band": row["Band"],
                    "base_salary": row["Base Salary (INR)"],
                    "performance_bonus": row["Performance Bonus (INR)"],
                    "retention_bonus": row["Retention Bonus (INR)"],
                    "ctc": row["Total CTC (INR)"],
                    "joining_date": row["Joining Date"],
                    "location": row["Location"],
                    # Fallbacks — use Department if not explicitly present
                    "function": row.get("Function", row["Department"]),
                    "team": row.get("Team", row["Department"]),
                }
    raise ValueError(f"No employee named '{name}' found in metadata.")

# 🧠 Load Chroma vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_settings = Settings(
    persist_directory="vectorstore/chroma_db",
    anonymized_telemetry=False
)
vectordb = Chroma(
    embedding_function=embedding_model,
    client_settings=chroma_settings,
    collection_name="offer_letters"
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 🧱 LangGraph state type
class State(dict):
    """Graph state format: {'name': str, 'employee_data': dict, 'docs': list, 'letter': str}"""
    pass

# 🔹 Node 1 – Load employee data
def get_employee_node(state: State):
    if "name" not in state:
        raise ValueError("❌ Missing key 'name' in state.")
    print("✅ Getting employee data for:", state["name"])
    employee_data = load_employee_data(state["name"])
    return {**state, "employee_data": employee_data}

# 🔹 Node 2 – Retrieve policy documents
def retrieve_docs_node(state: State):
    print("📄 Retrieving docs for:", state["name"])
    docs = retriever.invoke(f"Generate offer letter for {state['name']}")
    return {**state, "docs": docs}

# 🔹 Node 3 – Generate the offer letter
def generate_letter_node(state: State):
    employee = state["employee_data"]
    context = "\n\n".join(doc.page_content for doc in state["docs"])

    prompt = PromptTemplate.from_template("""
You are an HR assistant at Company ABC.

Write a detailed, personalized offer letter based on:
- The employee's role, band, salary, and joining date
- Relevant company policies (leave, travel, WFO)
- Formal tone and structure like the sample letter

Employee Info:
Name: {name}
Position: {position}
Band: {band}
CTC: {ctc}
Joining Date: {joining_date}
Function: {function}
Team: {team}
Location: {location}

Context (policies & samples):
{context}

Output only the full formatted offer letter.
""")

    full_prompt = prompt.format(
        name=employee["name"],
        position=employee["position"],
        band=employee["band"],
        ctc=employee["ctc"],
        joining_date=employee["joining_date"],
        function=employee["function"],
        team=employee["team"],
        location=employee["location"],
        context=context,
    )

    print("✍️ Generating offer letter...")
    letter = llm.invoke(full_prompt).content
    return {**state, "letter": letter}

# 🔄 Build the LangGraph
graph = StateGraph(State)
graph.add_node("load_employee", RunnableLambda(get_employee_node))
graph.add_node("retrieve_docs", RunnableLambda(retrieve_docs_node))
graph.add_node("generate_letter", RunnableLambda(generate_letter_node))

graph.set_entry_point("load_employee")
graph.add_edge("load_employee", "retrieve_docs")
graph.add_edge("retrieve_docs", "generate_letter")
graph.add_edge("generate_letter", END)

# ✅ Final app
app = graph.compile()
