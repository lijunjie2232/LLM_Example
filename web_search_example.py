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
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from baidusearch.baidusearch import search
import pickle

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""

MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"

model = ChatOllama(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)


prompt = (
    "You have access to a tool that retrieves context from official documents. "
    "Use the tool to help answer user queries."
)


def get_urls(results):
    if not results:
        return []
    urls = [result["url"] for result in results]
    for idx, url in enumerate(urls):
        if url.startswith("/"):
            urls[idx] = f"https://www.baidu.com{url}"
        if url.startswith("https"):
            urls[idx] = f"https:{url[6:]}"
    return urls


@tool("baidu_search")  # Custom name
def get_search_results(keyword, num_results=3):
    """seach baidu for keyword"""
    results = search(keyword, num_results=num_results)
    assert results
    documents = WebBaseLoader(get_urls(results)).load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=0
    ).split_documents(documents)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in documents
    )
    return serialized


parser = StrOutputParser()


class CustomAgentState(AgentState):
    user_id: str
    session_id: str


from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=[get_search_results],
    system_prompt=prompt,
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )
    ],
)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
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
                {
                    "messages": [{"role": "user", "content": request.msg}],
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                },
                {
                    "configurable": {
                        "thread_id": request.session_id,
                    }
                },
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
