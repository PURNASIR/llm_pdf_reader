import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(title="Dynamic PDF Embed & Query API", version="2.0")

# Allow CORS for Postman / frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Endpoint 1: Upload PDF + Embed + Save to Chroma
# ---------------------------------------------------------
@app.post("/upload_and_embed")
async def upload_and_embed(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    """
    Upload a PDF (file) and JSON (containing folder paths) in the same form-data body.
    Example JSON:
    {
      "pdf_folder": "E:/AIMODEL/input",
      "output_folder": "E:/AIMODEL/output"
    }
    """
    try:
        # --- Parse JSON input ---
        try:
            data = json.loads(json_data)
            pdf_folder = data.get("pdf_folder")
            output_folder = data.get("output_folder")

            if not pdf_folder or not output_folder:
                return JSONResponse(
                    {"error": "Both 'pdf_folder' and 'output_folder' are required."},
                    status_code=400
                )
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON format in 'json_data' field."}, status_code=400)

        # --- Ensure folders exist ---
        os.makedirs(pdf_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        # --- Save uploaded PDF ---
        filename = file.filename
        file_path = os.path.join(pdf_folder, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"📘 PDF saved at: {file_path}")

        # --- Load and process PDF ---
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return JSONResponse({"error": "No readable text found in the PDF."}, status_code=400)

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = [t for t in splitter.split_documents(docs) if t.page_content.strip()]

        if not texts:
            return JSONResponse({"error": "No valid text chunks found."}, status_code=400)

        # --- Create embeddings ---
        print("🔍 Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=output_folder)
        vectorstore.persist()

        print(f"✅ Chroma DB saved at: {output_folder}")

        return {
            "message": "✅ PDF processed successfully!",
            "pdf_file_path": file_path,
            "chroma_output_path": output_folder,
            "chunks_created": len(texts)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------
# Endpoint 2: Query from Existing Chroma DB
# ---------------------------------------------------------
@app.post("/query_chroma")
async def query_chroma(
    output_folder: str = Form(...),
    question: str = Form(...)
):
    """
    Query an existing Chroma DB (without re-uploading PDF).
    Example form-data:
    - output_folder: E:/AIMODEL/output
    - question: What is this PDF about?
    """
    try:
        # --- Validate Chroma DB path ---
        if not os.path.exists(output_folder):
            return JSONResponse({"error": f"Output folder '{output_folder}' not found."}, status_code=400)

        print(f"📂 Loading Chroma DB from: {output_folder}")

        # --- Load embeddings and Chroma DB ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=output_folder, embedding_function=embeddings)

        # --- Create retriever & small local LLM ---
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        nlp_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)
        llm = HuggingFacePipeline(pipeline=nlp_pipeline)

        # --- Build QA Chain ---
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        print(f"🧠 Query: {question}")
        result = qa_chain.invoke({"query": question})

        return {
            "question": question,
            "answer": result["result"],
            "source": output_folder
        }

    except Exception as e:
        print(f"❌ Error querying Chroma DB: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------
# Root endpoint for quick health check
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🚀 Dynamic PDF Embed & Query API is running!",
        "endpoints": {
            "/upload_and_embed": "Upload PDF and create Chroma DB",
            "/query_chroma": "Query existing Chroma DB by folder path"
        }
    }
