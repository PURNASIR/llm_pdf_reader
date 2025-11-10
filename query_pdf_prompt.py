import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --- Paths ---
DB_PATH = r"E:\AgenticAIWorkspace\llm-pdf-reader\pdfoutputdb\db"

# --- Load embeddings & Chroma vectorstore ---
print("📦 Loading Chroma vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# --- Create retriever (this is where the splitter’s chunks are retrieved) ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("🔍 Retriever ready. Fetching top 3 matching text chunks per query.\n")

# --- Initialize a small, local HuggingFace LLM (you can change to OpenAI if you have API key) ---
print("🤖 Loading local mini LLM for prompt-based response...")
model_name = "distilbert-base-uncased"  # small transformer, for local testing
nlp_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)

llm = HuggingFacePipeline(pipeline=nlp_pipeline)

# --- Create Retrieval-QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # Combines retrieved chunks into one prompt
)

# --- Query loop ---
print("\n💬 Type your question (or 'exit' to quit):\n")
while True:
    query = input("🧠 Ask: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting. Goodbye!")
        break

    result = qa_chain.invoke({"query": query})
    print("\n🗣️ Answer:")
    print(result["result"])
    print("\n" + "-" * 80 + "\n")
