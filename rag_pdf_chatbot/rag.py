import os

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


# --------------------------------
# 1. Environment variables load
# --------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


# --------------------------------
# 2. Groq client
# --------------------------------

groq_client = Groq(
    api_key=api_key
)


# --------------------------------
# 3. Embedding model
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 4. ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="documents"
)


# --------------------------------
# 5. Retrieve relevant documents
# --------------------------------

def retrieve_documents(question, number_of_results=3):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]

    return documents


# --------------------------------
# 6. Generate answer using LLM
# --------------------------------

def generate_answer(question, documents):

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say: "I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer


# --------------------------------
# 7. Main RAG pipeline
# --------------------------------

question = input(
    "Ask a question about your PDF: "
)

# Retrieval
documents = retrieve_documents(question)

# Generation
answer = generate_answer(
    question,
    documents
)


# --------------------------------
# 8. Display result
# --------------------------------

print("\n" + "=" * 60)

print("ANSWER:")

print(answer)

print("=" * 60)