import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain

# Temperature 설정
openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

# 프롬프트 템플릿 설정
prompt1 = PromptTemplate(
    input_variables=['review'],
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}"
)
chain1 = LLMChain(llm=openai, prompt=prompt1, output_key="translation")

prompt2 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)
chain2 = LLMChain(llm=openai, prompt=prompt2, output_key="summary")

prompt3 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. 숫자만 대답하세요.\n\n{translation}"
)
chain3 = LLMChain(llm=openai, prompt=prompt3, output_key="sentiment_score")

prompt4 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요.\n\n{review}"
)
chain4 = LLMChain(llm=openai, prompt=prompt4, output_key="language")

prompt5 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요.\n답변 언어:{language}\n리뷰 요약:{summary}"
)
chain5 = LLMChain(llm=openai, prompt=prompt5, output_key="reply1")

prompt6 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한국어로 번역해주세요. \n 리뷰 번역 {reply1}"
)

chain6 = LLMChain(llm=openai, prompt=prompt6, output_key="reply2")

# 체인 설정
all_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4, chain5,chain6],
    input_variables=['review'],
    output_variables=['translation', 'summary', 'sentiment_score', 'language', 'reply1','reply2'],
)

# 숙박 시설 리뷰 입력
review = """
The hotel was clean and the staff were very helpful. 
The location was convenient, close to many attractions. 
However, the room was a bit small and the breakfast options were limited. 
Overall, a decent stay but there is room for improvement.
"""

# 체인 실행 및 결과 출력
try:
    result = all_chain.invoke(input={'review': review})
    print(f'translation 결과: {result["translation"]} \n')
    print(f'summary 결과: {result["summary"]} \n')
    print(f'sentiment_score 결과: {result["sentiment_score"]} \n')
    print(f'language 결과: {result["language"]} \n')
    print(f'reply 결과: {result["reply1"]} \n')
    print(f'reply 결과: {result["reply2"]} \n')
except Exception as e:
    print(f"Error: {e}")