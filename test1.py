import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-pt",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

model = HuggingFacePipeline(pipeline=pipe)

if __name__ == "__main__":
    chain = model

    msg = [
        SystemMessage(content="translate following sentence into Chinese"),
        HumanMessage(content="where is the bath room?"),
    ]

    result = model.invoke(msg)

    print(result)

    pass
