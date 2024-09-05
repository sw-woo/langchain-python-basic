import streamlit as st
from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
# C Transformers는 Llama, GPT4All-J, MPT, Falcon과 같은 다양한 오픈 소스 모델을 지원합니다.
from langchain_community.llms.ctransformers import CTransformers
# ollama llama3.1model 연결하기
from langchain_ollama.llms import OllamaLLM

def getLLMResponse(form_input, email_sender, email_recipient, language):
    """
    getLLMResponse 함수는 주어진 입력을 사용하여 LLM(대형 언어 모델)으로부터 이메일 응답을 생성합니다.

    매개변수:
    - form_input: 사용자가 입력한 이메일 주제.
    - email_sender: 이메일을 보낸 사람의 이름.
    - email_recipient: 이메일을 받는 사람의 이름.
    - language: 이메일이 생성될 언어 (한국어 또는 영어).

    반환값:
    - LLM이 생성한 이메일 응답 텍스트.
    """

    # Llama-2-7B-Chat용 래퍼: CPU에서 Llama 2 실행

    # C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library
    # 만약 컴퓨터 사양에 맞추어서 모델을 local 컴퓨터에 다운받고 model="다운받은 모델명"을 작성해서 진행하시면 됩니다. 용량이 작을수록 경량화 된 버전이여서 성능이 조금 떨어질 수 있습니다.

    # GPT와 같은 언어 모델에서는 이러한 파일 형식을 사용하여 모델의 학습된 가중치와 구조를 저장하고, 이를 추론(inference) 시 불러와서 사용합니다.
    # 이러한 파일 형식은 모델을 효율적으로 저장하고 불러올 수 있게 하여, 다양한 플랫폼과 환경에서 모델 추론을 원활하게 할 수 있게 합니다.
    # 따라서, GGUF 및 GGML 파일 형식은 GPT와 같은 언어 모델의 맥락에서 모델 추론을 위해 사용되는 중요한 파일 형식입니다

    # llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',  # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
    #                     model_type='llama',
    #                     config={'max_new_tokens': 512,
    #                             'temperature': 0.01})

    # ollama llama3.1 부분 연결
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)

    if language == "한국어":
        template = """ 
        {email_topic} 주제를 포함한 이메일을 작성해 주세요.\n\n보낸 사람: {sender}\n받는 사람: {recipient} 전부 {language}로 번역해서 작성해주세요. 한문은 내용에서 제외해주세요.
        \n\n이메일 내용:
        """
    else: 
        template = """ 
        Write an email including the topic {email_topic}.\n\nSender: {sender}\nRecipient: {recipient} Please write the entire email in {language}.\n\nEmail content:
        """

    # 최종 PROMPT 생성
    prompt = PromptTemplate(
        input_variables=["email_topic", "sender", "recipient", "language"],
        template=template,
    )

    # LLM을 사용하여 응답 생성
    response = llm.invoke(prompt.format(email_topic=form_input, sender=email_sender, recipient=email_recipient, language=language))
    print(response)

    return response


st.set_page_config(
    page_title="이메일 생성기 📮",
    page_icon='📮',
    layout='centered',
    initial_sidebar_state='collapsed'
)
st.header("이메일 생성기 📮 ")

# 이메일 작성 언어 선택 
language_choice = st.selectbox('이메일을 작성할 언어를 선택하세요:', ['한국어', 'English'])

form_input = st.text_area('이메일 주제를 입력하세요', height=100)

# 사용자 입력을 받기 위한 UI 열 생성
col1, col2 = st.columns([10, 10])
with col1:
    email_sender = st.text_input('보낸 사람 이름')
with col2:
    email_recipient = st.text_input('받는 사람 이름')

submit = st.button("생성하기")

# '생성하기' 버튼이 클릭되면, 아래 코드를 실행합니다.
if submit:
    with st.spinner('생성 중입니다...'):
        response = getLLMResponse(form_input, email_sender, email_recipient, language_choice)
        st.write(response)
