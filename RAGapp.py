from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -------------------------
# Load API Key
# -------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found!")

# -------------------------
# Load PDF
# -------------------------
pdf_path = "data.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ File not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"📄 Loaded documents: {len(documents)}")

if not documents:
    raise ValueError("❌ No content found in PDF. Try another file.")

# -------------------------
# Split text
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"✂️ Chunks created: {len(docs)}")

if not docs:
    raise ValueError("❌ No chunks created. PDF may be empty or image-based.")

# -------------------------
# Embeddings
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------------
# Create / Load FAISS index
# -------------------------
index_path = "faiss_index"

if os.path.exists(index_path):
    print("📦 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("⚡ Creating new FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print("✅ FAISS index saved!")

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

print("\n🤖 RAG Chatbot Ready (type 'exit' to quit)\n")

# -------------------------
# Chat loop
# -------------------------
while True:
    query = input("Ask question: ")

    if query.lower() == "exit":
        break

    try:
        # Retrieve relevant chunks
        retrieved_docs = vectorstore.similarity_search(query, k=3)

        if not retrieved_docs:
            print("⚠️ No relevant context found.\n")
            continue

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # LLM response
        response = llm.invoke([
            SystemMessage(content="Answer ONLY from the provided context. If not found, say 'I don't know'."),
            HumanMessage(content=f"""
Context:
{context}

Question:
{query}
""")
        ])

        print("\n📌 Answer:\n")
        print(response.content)
        print("\n" + "-"*60 + "\n")

    except Exception as e:
        print("❌ Error:", str(e))