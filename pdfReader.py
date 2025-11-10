import os
import warnings
from shutil import which

warnings.filterwarnings("ignore", category=SyntaxWarning)

from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Input & Output paths ---
PDF_INPUT_FOLDER = r"E:\AgenticAIWorkspace\llm-pdf-reader\pdffiles"
OUTPUT_DB_FOLDER = r"E:\AgenticAIWorkspace\llm-pdf-reader\pdfoutputdb\db"

# --- Ensure output folder exists ---
os.makedirs(OUTPUT_DB_FOLDER, exist_ok=True)

print(f"📂 Reading PDFs from: {PDF_INPUT_FOLDER}")
print(f"💾 Chroma DB output folder: {OUTPUT_DB_FOLDER}\n")

# --- Utility check functions ---
def has_poppler() -> bool:
    """Check if Poppler is installed."""
    return which("pdftoppm") is not None

def has_tesseract() -> bool:
    """Check if Tesseract OCR is installed."""
    return which("tesseract") is not None

# --- Detect OCR Tools ---
poppler_available = has_poppler()
tesseract_available = has_tesseract()

if poppler_available:
    print("✅ Poppler detected.")
else:
    print("⚠️ Poppler not found — OCR-based PDFs will be skipped.")

if tesseract_available:
    print("✅ Tesseract detected. OCR for scanned PDFs enabled.\n")
else:
    print("⚠️ Tesseract not found — image-based PDFs will be skipped.\n")

# --- Load documents safely ---
docs = []
for filename in os.listdir(PDF_INPUT_FOLDER):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_INPUT_FOLDER, filename)
    try:
        # Use OCR loader only if both Poppler and Tesseract are available
        if poppler_available and tesseract_available:
            loader = UnstructuredFileLoader(pdf_path)
        else:
            loader = PyPDFLoader(pdf_path)

        loaded_docs = loader.load()

        if not loaded_docs:
            print(f"⚠️ {filename} returned no text, skipping.")
            continue

        docs.extend(loaded_docs)
        print(f"✅ Loaded: {filename} ({len(loaded_docs)} sections)")

    except Exception as e:
        print(f"⚠️ Skipped {filename} due to error: {e}")

print(f"\n📄 Total valid documents loaded: {len(docs)}")

# --- Split text into chunks ---
if not docs:
    print("❌ No readable PDFs found. Nothing to embed.")
    exit(0)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_documents(docs)

# Remove empty chunks
texts = [t for t in texts if t.page_content.strip()]
print(f"✂️ Split into {len(texts)} non-empty text chunks\n")

if not texts:
    print("⚠️ No valid text chunks found after filtering. Nothing to embed.")
    exit(0)

# --- Print preview of chunks ---
print("📖 Preview of extracted text chunks:\n")
for i, chunk in enumerate(texts[:5]):  # print first 5 chunks for verification
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content[:500])  # show first 500 chars
    print("---------------------------\n")

# --- Create embeddings ---
print("🔍 Generating HuggingFace embeddings (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Create and persist Chroma DB ---
print("💾 Creating Chroma vector store...")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=OUTPUT_DB_FOLDER)
vectorstore.persist()  # Save to disk

print(f"\n✅ All done! PDF embeddings stored successfully in:\n{OUTPUT_DB_FOLDER}")
print("🎯 You can now query the Chroma DB or build a retriever on top of it.")
