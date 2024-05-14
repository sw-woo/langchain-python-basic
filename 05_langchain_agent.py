#pip install wikipedia
#pip install arxiv
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

#agent tools 중 wikipedia 사용
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

#langchain 공식 문서 검색을 위한 검색기 역할을 하는 벡터 DB 생성 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 벡터 DB를 agent에게 전달하기 위한 tool생성 
from langchain.tools.retriever import create_retriever_tool

#arxiv 논문 검색을 위한 tool 생성
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

#pip install yfinance
#yfinance api를 사용하기 위한 tool 생성
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


#agent 시물레이션을 위한 prompt 참조
from langchain import hub

#Agents 생성을 위한 참조 
from langchain.agents import create_openai_tools_agent

# Agent Executer 
from langchain.agents import AgentExecutor

#pip install langchainhub
#hub에서 가져온 prompt를 agent에게 전달하기 위한 prompt 생성
# prompt = hub.pull("sungwoo/openai-fuctions-ko-agent")
prompt = hub.pull("hwchase17/openai-functions-agent")

# 일관된 값을 위하여 Temperature 0.1로 설정 model은 gpt-4o로 설정
openai = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)

#Wikipedia API 설정 : top_k_results = 결과 수, doc_content_chars_max = 문서 길이 제한
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)

wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki.name)


#네이버 기사 내용을 가져와서 벡터 DB 생성
loader=WebBaseLoader("https://news.naver.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()

print(retriever)

retriever_tool = create_retriever_tool(retriever,"naver_news_search","네이버 뉴스정보가 담김 벡터 DB 당일 기사에 대해서 궁금하면 이 툴을 사용하세요!")

print(retriever_tool.name)


#arxiv tool
#Wikipedia API 설정 : top_k_results = 결과 수, doc_content_chars_max = 문서 길이 제한
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

print(arxiv.name)

#Yahoo Finance News API 설정 : top_k=결과 수
yfinace = YahooFinanceNewsTool(top_k=2)

# agent가 사용할 tool을 정의하여 tools에 저장
tools=[wiki,retriever_tool,arxiv,yfinace]

#agent llm 모델을 openai로 정의하고 tools ,prompt를 입력하여 agent를 완성한다.
agent = create_openai_tools_agent(llm=openai,tools=tools,prompt=prompt)

#agent Execute 정의 부분 

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

#최종적으로 agent_executor를 실행하여 결과를 출력한다.

agent_result = agent_executor.invoke({"input":"마이크로소프트 관련 오늘자 소식을 알려줘!"})
print(agent_result)












