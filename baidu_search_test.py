from baidusearch.baidusearch import search
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


results = search("今天上海的天气如何", num_results=3)
assert results
documents = WebBaseLoader(get_urls(results)).load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=2048, chunk_overlap=0
).split_documents(documents)
serialized = "\n\n".join(
    (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in documents
)
pass
