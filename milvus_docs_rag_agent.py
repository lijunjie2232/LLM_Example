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
from langchain.tools import tool
from langchain.agents import create_agent
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


retriever = vector_store.as_retriever()


prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)


parser = StrOutputParser()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str, topk=5):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=topk)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt=prompt,
)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

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
async def chat(request: ChatRequest):
    try:

        def stream_handler():
            for event in agent.stream(
                {"messages": [{"role": "user", "content": request.msg}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
                yield f"{event['messages'][-1].content} \n\n"

        return StreamingResponse(stream_handler(), media_type="text/plain")
    except Exception as e:
        print(e)
        return str(e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
