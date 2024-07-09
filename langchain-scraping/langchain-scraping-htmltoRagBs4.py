from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(
    class_=("sa_text", "sa_item _SECTION_HEADLINE"))
loader = WebBaseLoader(
    web_paths=("https://news.naver.com/section/101",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# print(docs)

print(len(docs[0].page_content))
# print(print(docs[0].page_content[:500]))

# Split data into chunks of small chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)

len(all_splits[0].page_content)

print(f"시작단어 진행하기 {all_splits[10].metadata["start_index"]}")


vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 1})

retrieved_docs = retriever.invoke(
    "오늘 우리은행 대출 관련해서 중요한 부분만 알려줘!")


# print(retrieved_docs[0].page_content.replace('\n', ''))

print("------------")

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

template = """
다음 문맥만을 토대로 질문에 답하세요.
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content.replace('\n', '') for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("오늘 우리은행 대출 관련해서 중요한 부분만 알려줘!")
print(result)

# example_message = prompt.invoke(
#     {"context": retrieved_docs[0].page_content.replace('\n', ''),
#         "question": "오늘 우리은행 대출 관련해서 중요한 부분만 알려줘!"}
# ).to_messages()

# print(example_message[0].content)

# Post-processing
