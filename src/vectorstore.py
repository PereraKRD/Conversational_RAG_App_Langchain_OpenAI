import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.helper import load_pdf_files, process_documents
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings( api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def initialize_faiss_vectorstore(splits):
    return FAISS.from_texts([t.page_content for t in splits], embedding=embedding_model)

pdf_folder_path = "Docs/"
extracted_documents = load_pdf_files(pdf_folder_path)
splits = process_documents(extracted_documents)
vectorstore = initialize_faiss_vectorstore(splits)