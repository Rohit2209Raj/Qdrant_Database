# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType
)

import os
import json


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# QDRANT CONNECTION
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")


# ============================================================
# COLLECTION CONFIGURATION
# ============================================================

COLLECTION_NAME = "knowledge_langchain"
EMBEDDING_SIZE = 384


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# CREATE COLLECTION ONLY IF IT DOES NOT EXIST
# ============================================================

# if not qdrant_client.collection_exists(COLLECTION_NAME):

# print(f"Collection does not exist.")

    # --------------------------------------------------------
    # CREATE COLLECTION
    # --------------------------------------------------------

if qdrant_client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    qdrant_client.delete_collection(COLLECTION_NAME)


qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

print(f"Created collection: {COLLECTION_NAME}")

    # --------------------------------------------------------
    # CREATE PAYLOAD INDEX
    # --------------------------------------------------------

qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.category",
        field_schema=PayloadSchemaType.KEYWORD
    )

print("Created payload index: metadata.category")

    # --------------------------------------------------------
    # LOAD KNOWLEDGE
    # --------------------------------------------------------

with open("knowledge.json", "r", encoding="utf-8") as f:
        data = json.load(f)

print(f"Loaded {len(data)} documents")

    # --------------------------------------------------------
    # CONVERT TO LANGCHAIN DOCUMENTS
    # --------------------------------------------------------

documents = [
        Document(
            page_content=item["text"],
            metadata={
                "category": item["category"]
            }
        )
        for item in data
    ]

print("Converted data into LangChain Documents")

    # --------------------------------------------------------
    # CONNECT LANGCHAIN WITH QDRANT
    # --------------------------------------------------------

db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model
    )

    # --------------------------------------------------------
    # STORE DOCUMENTS
    # --------------------------------------------------------

db.add_documents(documents)

print(f"Stored {len(documents)} documents in Qdrant")


# ============================================================
# IF COLLECTION ALREADY EXISTS
# ============================================================

# else:

#     print(f"Collection already exists: {COLLECTION_NAME}")
#     print("Connecting to existing Qdrant collection...")

#     db = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=COLLECTION_NAME,
#         embedding=embeddings_model
#     )

#     print("Connected to existing vector database")


# # ============================================================
# # LLM
# # ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)


# ============================================================
# BASIC RETRIEVER
# ============================================================

retriever = db.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# USER QUERY
# ============================================================

query = input("\nEnter your query: ")

print(f"\nOriginal Query: {query}")


# ============================================================
# NORMAL SEMANTIC SEARCH
# ============================================================

docs = retriever.invoke(query)


print("\n================ RETRIEVED DOCUMENTS ================")

for i, doc in enumerate(docs, start=1):

    print(f"\nDocument {i}")

    print(
        f"Category: "
        f"{doc.metadata.get('category')}"
    )

    print(
        f"Content: "
        f"{doc.page_content}"
    )


# ============================================================
# FILTERED RETRIEVER
# ============================================================

filtered_retriever = db.as_retriever(
    search_kwargs={
        "k": 3,

        "filter": {
            "must": [
                {
                    "key": "metadata.category",

                    "match": {
                        "value": "reimbursement"
                    }
                }
            ]
        }
    }
)


# ============================================================
# FILTERED SEARCH
# ============================================================

filtered_docs = filtered_retriever.invoke(query)


print("\n================ FILTERED RESULTS ================")

for i, doc in enumerate(filtered_docs, start=1):

    print(f"\nDocument {i}")

    print(
        f"Category: "
        f"{doc.metadata.get('category')}"
    )

    print(
        f"Content: "
        f"{doc.page_content}"
    )


# ============================================================
# CREATE CONTEXT
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ============================================================
# RAG PROMPT
# ============================================================

prompt = f"""
Answer the question using ONLY the information provided
in the context.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say:

"I don't know based on the provided information."
"""


# ============================================================
# LLM RESPONSE
# ============================================================

response = llm.invoke(prompt)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\n================ FINAL ANSWER ================")

print(response.content)