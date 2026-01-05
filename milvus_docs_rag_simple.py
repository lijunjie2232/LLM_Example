import os

from uuid import uuid4
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    RunnableConfig,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
import pickle

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""

MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"

model = ChatOllama(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)

milvus_documents = pickle.load(open("milvus_documents.pkl", "rb"))

VCDB_URI = "./milvus_docs.db"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=0)

all_splits = text_splitter.split_documents(milvus_documents)

connection_args = {"uri": VCDB_URI}

COLLECTION_NAME = "MilvusDocs"

# vector_store = Milvus(
#     embedding_function=embeddings,
#     collection_name=COLLECTION_NAME,
#     connection_args=connection_args,
#     drop_old=True,
# ).from_documents(
#     embedding=embeddings,
#     documents=all_splits,
#     index_params={"index_type": "FLAT", "metric_type": "L2"},
#     collection_name=COLLECTION_NAME,
#     connection_args=connection_args,
# )

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args=connection_args,
    drop_old=True,
)

# vector_store.similarity_search_with_score(
#     query="cors",
# )

retriever = vector_store.as_retriever()


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that can answer questions about the universe."
            + "Use the following pieces of context to answer the question at the end."
            + "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            + "Use three sentences maximum and keep the answer as concise as possible"
            + """ and say "thanks for asking!" at the end of the answer.""",
        ),
        (
            "system",
            "The context is as follow:\n {context}",
        ),
        # (
        #     "system",
        #     "The history is as follow:\n {history}",
        # ),
        (
            "system",
            "The Question is as follows:",
        ),
        ("human", "{msg}"),
    ]
)

parser = StrOutputParser()

rag_chain = (
    {
        "context": retriever,
        "msg": RunnablePassthrough(),
    }
    | rag_prompt
    | model
    | parser
)


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    msg: str


class ChatResponse(BaseModel):
    code: int
    msg: str


class NewSessionResponse(BaseModel):
    session_id: str


@app.post("/api/v1/new-session")
def create_session() -> NewSessionResponse:
    session_id = str(uuid4())
    return NewSessionResponse(session_id=session_id)


@app.post("/api/v1/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(
            code=200,
            msg=rag_chain.invoke(
                request.msg,
            ),
        )
    except Exception as e:
        print(e)
        return ChatResponse(code=500, msg=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
