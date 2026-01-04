import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""

model = ChatOllama(
    model="gpt-oss:20b",
    base_url="http://127.0.0.1:11434",
)

msg = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Chinese."
    ),
    HumanMessage(content="I love programming."),
]

# result = model.invoke(msg)
parser = StrOutputParser()

chain = model | parser

result = chain.invoke(msg)

print(result)

pass
