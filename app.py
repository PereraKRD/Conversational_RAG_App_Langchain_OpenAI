import os
from fastapi import FastAPI, Request,HTTPException
from pydantic import BaseModel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from dotenv import load_dotenv
from src.llm import rag_chain

load_dotenv()

LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class QueryModel(BaseModel):
    session_id: str
    input: str

app = FastAPI(redoc_url=None, docs_url=None)

set_llm_cache(InMemoryCache())

allowed_origins = [
    "*"
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.get("/")
async def read_root():
    try:
        return "Hello World !!!"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
def get_session_history_by_sessionid(session_id: str):
    try:
        return get_session_history(session_id)
    except Exception:
        if session_id not in store:
            raise HTTPException(status_code=404, detail="Session ID not found")
        else:
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/history/{session_id}")
def delete_session_history_by_sessionid(session_id: str):
    try:
        if session_id in store:
            del store[session_id]
        return "Deleted"
    except Exception:
        if session_id not in store:
            raise HTTPException(status_code=404, detail="Session ID not found")
        else:
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/query")
async def query(request: Request, query: QueryModel):
    try:
        response = conversational_rag_chain.invoke(
            {"input": query.input},
            config={"configurable": {"session_id": query.session_id}},
        )
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






#Exeception handeling middleware
# async def catch_exceptions_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception:
#         # you probably want some kind of logging here
#         print_exception(e)
#         return Response("Internal server error", status_code=500)

# app.middleware('http')(catch_exceptions_middleware)


# url_loader = UnstructuredURLLoader(
    #     urls=["https://galleryhr.com/", "https://galleryhr.com/about/"]
    # )
    # data = url_loader.load()
    # documents.extend(data)


    

# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pymongo import MongoClient


    # def initialize_mongo_client(client_uri):
#     client = MongoClient(client_uri)
#     db_name="langchain_db"
#     collection_name="GHR"
#     vector_search_index = "vector_index"
#     atlas_collection = client[db_name][collection_name]

#     return atlas_collection, vector_search_index

# atlas_collection, vector_search_index = initialize_mongo_client(MONGO_DB_URI)

# def initialize_mongo_vectorstore(splits):
#     return MongoDBAtlasVectorSearch.from_documents(
#     documents = splits,
#     embedding = OpenAIEmbeddings(disallowed_special=()),
#     collection = atlas_collection,
#     index_name = vector_search_index,
# )