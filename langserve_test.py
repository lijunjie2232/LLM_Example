from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langserve import add_routes

model = OllamaLLM(
    model="qw3_4b_i_2507:latest",
    base_url="http://127.0.0.1:11434",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant that translates {input_language} to {output_language}."
        ),
        HumanMessage(
            "Translate the following text from {input_language} to {output_language}: {text}"
        ),
    ]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

if __name__ == "__main__":
    app = FastAPI()

    add_routes(
        app,
        chain,
        path="/mychain",
    )
