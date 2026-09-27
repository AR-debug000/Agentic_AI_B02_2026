import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. PDF se text extract karna
# --------------------------------

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:
        text = page.get_text()
        full_text += text + "\n"

    doc.close()

    return full_text


# --------------------------------
# 2. Text ko chunks mein divide karna
# --------------------------------

def create_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


# --------------------------------
# 3. Main program
# --------------------------------

pdf_folder = "documents"

pdf_files = [
    file for file in os.listdir(pdf_folder)
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    print("No PDF found inside documents folder.")
    exit()


pdf_path = os.path.join(pdf_folder, pdf_files[0])

print("Reading PDF:", pdf_path)

text = extract_text_from_pdf(pdf_path)

print("PDF text extracted successfully.")

chunks = create_chunks(text)

print("Total chunks:", len(chunks))


# --------------------------------
# 4. Embedding model
# --------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------
# 5. ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


# --------------------------------
# 6. Embeddings generate karna
# --------------------------------

print("Creating embeddings...")

embeddings = embedding_model.encode(
    chunks
).tolist()


# --------------------------------
# 7. ChromaDB mein save karna
# --------------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


print("Data successfully stored in ChromaDB.")

print("RAG ingestion completed!")