import os
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFaceEmbeddings
from langchain_community.document_loaders import   PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma


load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

model_RAG = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


document = PyPDFLoader("files\document.pdf")


text = document.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_documents(text)

embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



vector_store = Chroma(
    collection_name="my_pdf_collection",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)

vector_store.add_documents(chunks)

retriever  = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k":4,
    }
)



result = retriever.invoke("According to the document, what is the difference between supervised learning, unsupervised learning, and reinforcement learning")















