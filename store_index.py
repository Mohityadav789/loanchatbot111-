import tempfile
import pinecone
import os
from src.helper  import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
import pinecone
from pinecone import Pinecone
# from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import  ServerlessSpec
from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_pinecone import PineconeVectorStore
# from langchain_pinecone import PineconeVectorStore

# from dotenv import load_dotenv
# import os
# from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
# from pinecone import Pinecone
# from pinecone import ServerlessSpec 
# from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files("data/")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

# embeddings = download_hugging_face_embeddings()


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)
print(tempfile.gettempdir())

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

print("Pinecone installed successfully")
pc


index_name = "loan-chatbot2"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec =  ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk, 
    index_name=index_name,
    embedding=embedding
)


# from langchain_pinecone import PineconeVectorStore


