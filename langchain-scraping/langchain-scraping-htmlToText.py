from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.beautiful_soup_transformer import BeautifulSoupTransformer
from langchain.chains.openai_functions.extraction import create_extraction_chain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from datetime import datetime
import os

urls = ["https://news.naver.com/section/101"]


loader = AsyncChromiumLoader(urls)
html = loader.load()


bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["div"])

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
splits = splitter.split_documents(docs_transformed)

schema = {
    "properties": {
        "뉴스 제목": {"type": "string"},
        "언론사": {"type": "string"},
    },
    "required": ["뉴스 제목", "언론사"],
}

llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)


# content = "뉴스제목은 '갤럭시 S24 시리즈' 사전 개통 시작…주요 매장서 예약자 대기 행렬이다. 그리고 언론사는 데일리안이다."

# Directory setup based on the current date
date_today = datetime.now().strftime("%Y-%m-%d")
output_dir = os.path.join(os.getcwd(), "text")
os.makedirs(output_dir, exist_ok=True)

# 수집하고 저장하는 기능 함수


def extract_and_save(content: str, schema: dict, directory: str):
    extracted_content = create_extraction_chain(
        schema=schema, llm=llm).invoke(content)
    filename = f"{datetime.now().strftime('%H-%M-%S')}.txt"
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(str(extracted_content))
    return extracted_content

# print(f"Extracted: {extract(content=content, schema=schema)}")


extracted_contents = []
for split in tqdm(splits):
    extracted_content = extract_and_save(
        content=split.page_content, schema=schema, directory=output_dir)
    extracted_contents.extend(extracted_content)


print(f"Extracted: {extracted_content}")
