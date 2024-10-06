# 필요한 패키지 설치
# pip install wikipedia # wikipedia API를 사용하기 위한 패키지
# pip install arxiv # Arxiv 논문 검색을 위한 패키지
# pip install yfinance # Yahoo Finance에서 주식 뉴스 및 데이터를 검색하기 위한 패키지

# Agents 생성을 위한 참조 Agent Executer
from langchain.agents import AgentExecutor

# 벡터 DB를 agent에게 전달하기 위한 tool생성
from langchain.agents import create_openai_tools_agent
# langchainhub 에서 제공하는 prompt 사용
from langchain import hub


# yfinance api를 사용하기 위한 tool 생성
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

# arxiv 논문 검색을 위한 tool 생성
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

# 벡터 DB구축 및 검색 도구
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# langchain 공식 문서 검색을 위한 검색기 역할을 하는 벡터 DB 생성
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# agent tools 중 wikipedia 사용
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

#openAI LLM 설정
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# 일관된 값을 위하여 Temperature 0.1로 설정 model은 gpt-4o로도 설정 할 수 있습니다.
openai = ChatOpenAI(
    model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)


# agent 시물레이션을 위한 prompt 참조
# hub에서 가져온 prompt를 agent에게 전달하기 위한 prompt 생성
prompt = hub.pull("hwchase17/openai-functions-agent")


# Wikipedia API 설정 : top_k_results = 결과 수, doc_content_chars_max = 문서 길이 제한
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki.name)


# 네이버 기사 내용을 가져와서 벡터 DB 생성
loader = WebBaseLoader("https://news.naver.com/") # 네이버 뉴스 웹 페이지 로드
docs = loader.load() # 웹 문서 로드
# 문서를 1000자의 덩어리로 나누되, 각 덩어리의 200자 정도는 중첩되도록 설정
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200).split_documents(docs)

# 문서를 임베딩하고 FAISS 벡터 DB로 저장
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever() # 벡터 DB를 검색기로 변환

#검색기 객체 출력 확인
print(retriever)


# 검색 도구 생성
retriever_tool = create_retriever_tool(
    retriever, "naver_news_search", "네이버 뉴스정보가 저장된 벡터 DB 당일 기사에 대해서 궁금하면 이 툴을 사용하세요!")

#툴 이름 출력 확인
print(retriever_tool.name)


# arxiv tool
# Wikipedia API 설정 : top_k_results = 결과 수, doc_content_chars_max = 문서 길이 제한
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1, doc_content_chars_max=200, load_all_available_meta=False,)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#arxiv tool 이름 출력 확인
print(arxiv.name)

# Yahoo Finance News API 설정 : top_k=결과 수
yfinace = YahooFinanceNewsTool(top_k=2)

# agent가 사용할 tool을 정의하여 tools에 저장
tools = [wiki, retriever_tool, arxiv, yfinace]

# agent llm 모델을 openai로 정의하고 tools ,prompt를 입력하여 agent를 완성한다.
agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)


# agent Execute 정의 부분 verbose=True로 설정하면 agent 실행과정을 출력합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_result = agent_executor.invoke({"input": "llm 관련 최신 논문을 알려줘"})
agent_result = agent_executor.invoke({"input": "마이크로소프트 관련 오늘자 주가를 알려줘"})

#결과 출력
print(agent_result)
