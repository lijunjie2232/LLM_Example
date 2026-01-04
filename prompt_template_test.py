from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Configure model with specific parameters
model = OllamaLLM(
    model="gpt-oss:20b",
    # temperature=0.3,  # Lower temperature for more focused responses
    # top_p=0.9,  # Limit token selection
    # top_k=40,  # Limit vocabulary consideration
    # num_predict=100,  # Limit response length
    # repeat_penalty=1.2,  # Penalize repetition
)

prompt = PromptTemplate(
    input_variables=["text", "input_language", "output_language"],
    template="""Translate the following text from {input_language} to {output_language} and only ouptut the translated {output_language} result: 
    "{text}" """,
)

parser = StrOutputParser()
chain = prompt | model | parser

# input_str = "The sky is Blue and the grass is Green."
# result = chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "French",
#         "text": input_str,
#     }
# )
# print(result)
# result = chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "Chinese",
#         "text": input_str,
#     }
# )
# print(result)

from fastapi import FastAPI, Form

app = FastAPI()


@app.post("/api/v1/translate")
async def translate(
    origin_lang: str = Form(..., alias="ol"),
    target_lang: str = Form(..., alias="tl"),
    input_str: str = Form(..., alias="input", max_length=512,),
) -> dict:
    result = chain.invoke(
            {
                "input_language": origin_lang,
                "output_language": target_lang,
                "text": input_str,
            }
        )
    return {
        "code": 200,
        "message": "ok",
        "result": str(result)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
    )
