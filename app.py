from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os
import sqlite3
import re


load_dotenv()

print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

app = Flask(__name__, template_folder="templates")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# -----------------------------
# DATABASE SETUP
# -----------------------------

def init_db():
    conn = sqlite3.connect("leads.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        city TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


def save_lead(name, phone, city):
    conn = sqlite3.connect("leads.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO leads (name, phone, city) VALUES (?, ?, ?)",
        (name, phone, city)
    )

    conn.commit()
    conn.close()


def extract_lead(text):

    phone_match = re.findall(r'\d{10}', text)

    words = text.lower().split()

    name = None
    city = None
    phone = None

    if phone_match:
        phone = phone_match[0]

    if "i am" in text.lower():
        name = text.lower().split("i am")[-1].split()[0]

    if "from" in text.lower():
        city = text.lower().split("from")[-1].split()[0]

    return name, phone, city

# -----------------------------
# MEMORY (THIS FIXES YOUR ISSUE)
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="context",
    return_messages=True
)

embedding = download_embeddings()

index_name = "loan-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

chatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system", "Conversation history: {context}"),
        ("human", "{input}")
    ]
)

questopm_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, questopm_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
# msg = request.form["msg"]
def chat():

    msg = request.form["msg"]
    user_input = msg

    print("User:", user_input)

    # -----------------------------
    # LEAD DETECTION
    # -----------------------------
    name, phone, city = extract_lead(user_input)

    if phone:
        print("Lead detected:", name, phone, city)
        save_lead(name, phone, city)

    # -----------------------------
    # LOAD CHAT MEMORY
    # -----------------------------
    chat_history = memory.load_memory_variables({})["context"]

    response = rag_chain.invoke({
        "input": user_input,
        "context": chat_history
    })

    print("Raw response:", response)

    if isinstance(response, dict):
        answer = response.get("answer", "Sorry, I couldn't process that.")
    else:
        answer = str(response)

    print("Bot:", answer)

    memory.save_context(
        {"input": user_input},
        {"output": answer}
    )

    return str(answer)

@app.route("/leads")
def leads():

    import sqlite3

    conn = sqlite3.connect("leads.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM leads")
    data = cursor.fetchall()

    conn.close()

    return render_template("leads.html", leads=data)
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)