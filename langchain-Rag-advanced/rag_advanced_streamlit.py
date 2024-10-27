import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from streamlit_extras.buy_me_a_coffee import button
from langchain.load import dumps, loads
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore



# 'Buy me a coffee' 버튼 초기화 username에 각자 username을 입력해주세요!
button(username="swpheus14", floating=True, width=221)

# 제목과 지침 설정
st.title("ChatPDF with Multiquery+hybridSearch+RagFusion")
st.write("---")
st.write("PDF 파일을 업로드하고 내용을 기반으로 질문하세요.")

# OpenAI API 키 입력
openai_key = st.text_input('OpenAI API 키를 입력해 주세요!', type="password")

# GPT 모델 선택
model_choice = st.selectbox(
    '사용할 GPT 모델을 선택하세요:',
    ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']
)

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드해 주세요!", type=['pdf'])
st.write("---")

# PDF를 문서로 변환하는 함수
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 문서를 포맷하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 파일이 업로드되었는지 확인
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(pages)

    # 임베딩 및 faiss 설정
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # 임베딩 벡터의 차원 계산 ex)1536 차원
    embedding_dimension = len(OpenAIEmbeddings(openai_api_key=openai_key).embed_query("hello world"))

    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings(openai_api_key=openai_key).embed_query("hello world")))

    # 벡터 스토어 생성
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={} 
    )

    # 문서 청크를 벡터 스토어에 추가
    vectorstore.add_documents(documents=splits, ids=range(len(splits)))

    #FAISS 리트리버 생성
    faiss_retriever = vectorstore.as_retriever(search_type="mmr",  # MMR 알고리즘을 사용하여 검색
                                                # 상위 1개의 문서를 반환하지만, 고려할 문서는 4개로 설정
                                                search_kwargs={'k': 1, 'fetch_k': 4})

    # BM25 리트리버 설정
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    # 앙상블 리트리버 설정
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.2, 0.8]
    )

    # RAG-Fusion을 위한 쿼리 생성
    template = """
    당신은 AI 언어 모델 조수입니다. 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다. 
    사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다. 
    각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Reciprocal Rank Fusion 함수
    def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results[:top_n]

    # RAG-Fusion 체인 설정
    retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion

    # 최종 RAG 체인 설정
    template = """다음 맥락을 바탕으로 질문에 답변하세요:

    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=model_choice, temperature=0,
                     openai_api_key=openai_key)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 사용자 질문 입력
    st.header("PDF에 질문하세요!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기(ASK)'):
        with st.spinner('답변 생성 중...'):
            result = final_rag_chain.invoke(question)
            st.write(result)