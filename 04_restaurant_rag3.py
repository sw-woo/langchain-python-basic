import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
restaurants_text = os.path.join(current_dir, 'restaurants.txt')
restaurant_faiss = os.path.join(current_dir, "restaurant-faiss")


def create_faiss_index():

    # TextLoader를 사용하여 "restaurants.txt" 파일에서 텍스트를 로드합니다.
    loader = TextLoader(os.path.join(current_dir, "restaurants.txt"))

    documents = loader.load()

    # 텍스트를 300자 단위로 나누고, 연속된 청크 사이에 50자의 겹침을 두어 텍스트를 분할합니다.
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.create_documents(documents)

    # OpenAI API를 사용하여 임베딩을 생성합니다.
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # Faiss 인덱스를 생성하고 저장합니다.
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(restaurant_faiss)
    print("Faiss Index created and saved")


def load_faiss_index():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    load_db = FAISS.load_local(
        restaurant_faiss, embeddings, allow_dangerous_deserialization=True)

    return load_db


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(db, query):
    # OpenAI 언어 모델 초기화
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 사용자 정의 프롬프트 템플릿 생성
    prompt_template = """
    당신은 유능한 AI 비서입니다. 주어진 맥락 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해야 합니다.

    맥락: {context}

    질문: {question}

    답변을 작성할 때 다음 지침을 따르세요:
    1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변하세요.
    2. 맥락 정보에 없는 내용은 답변에 포함하지 마세요.
    3. 질문과 관련이 없는 정보는 제외하세요.
    4. 답변은 간결하고 명확하게 작성하세요.
    5. 불확실한 경우, "주어진 정보로는 정확한 답변을 드릴 수 없습니다."라고 말하세요.

    답변:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = (
        {
            "context": db.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 질문에 대한 답변 생성
    result = qa_chain.invoke(query)

    return result


def main():
    # FAISS 인덱스가 없으면 생성
    if not os.path.exists(restaurant_faiss):
        create_faiss_index()

    # Faiss 인덱스를 로드합니다.
    db = load_faiss_index()

    while True:
        query = input("레스토랑에 대해서 궁금한 점을 물어보세요 (종료하려면 'quit' 입력): ")

        if query.lower() == 'quit':
            break

        answer = answer_question(db, query)
        print(f"답변: {answer}\n")


if __name__ == "__main__":
    main()
