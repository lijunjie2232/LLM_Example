from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory

from loguru import logger
from time import sleep

# Configure model with specific parameters
model = OllamaLLM(
    model="gpt-oss:20b",
    # temperature=0.3,  # Lower temperature for more focused responses
    # top_p=0.9,  # Limit token selection
    # top_k=40,  # Limit vocabulary consideration
    # num_predict=100,  # Limit response length
    # repeat_penalty=1.2,  # Penalize repetition
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that can answer questions about the universe.",
        ),
        MessagesPlaceholder(variable_name="msg"),
    ]
)

parser = StrOutputParser()
chain = prompt | model | parser

from uuid import uuid4

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="msg",
    output_key="text",
)


def get_runnable_config(session_id):
    return RunnableConfig(configurable={"session_id": session_id})


from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ChatRequest(BaseModel):
    msg: str


class NewSessionResponse(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    code: int
    msg: str


app = FastAPI()


@app.post("/api/v1/new-session")
def create_session() -> NewSessionResponse:
    session_id = str(uuid4())
    return NewSessionResponse(session_id=session_id)


@app.post("/api/v1/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    try:
        assert session_id, "Session ID is required"
        result = do_message.invoke(
            request.model_dump(),
            config=get_runnable_config(session_id),
        )
        return ChatResponse.model_validate({"code": 200, "msg": result})
    except Exception as e:
        print("Error:", e)
        return ChatResponse.model_validate({"code": 500, "msg": f"Error: {e}"})


# stream type of do_message
@app.post("/api/v1/chat-stream/{session_id}")
async def chat_stream(session_id: str, request: ChatRequest):

    try:
        assert session_id, "Session ID is required"

        def iter_stream():
            for result in do_message.stream(
                request.model_dump(),
                get_runnable_config(session_id),
            ):
                print(result, end="")
                yield result

        return StreamingResponse(
            iter_stream(),
            media_type="text/plain",
        )

    except Exception as e:
        print("Error:", e)
        return StreamingResponse(
            [f"Error: {e}"],
            media_type="text/plain",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
