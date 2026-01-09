from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
import pickle
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""

MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"

model = ChatOllama(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)


db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')


toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
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
