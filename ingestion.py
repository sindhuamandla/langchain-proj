import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    #print(os.environ['PINECONE_API_KEY'])

    loader = TextLoader(
    "C:/Users/rajir/Desktop/langchain-course/mediumblog1.txt",
    encoding="utf-8"
)
    document = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = PineconeVectorStore.from_documents(
        texts,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"]
    )

    print("finish")

    
