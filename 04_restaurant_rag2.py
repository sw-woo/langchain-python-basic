from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
import asyncio
load_dotenv()

# pip install faiss-cpu or faiss-gpu
current_dir = os.path.dirname(os.path.abspath(__file__))

# main()이라는 비동기 함수를 정의합니다.


async def main():
    # 환경 변수에서 가져온 OpenAI API 키를 사용하여 OpenAIEmbeddings 클래스를 초기화합니다.
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # 지정된 임베딩을 사용하여 로컬에 저장된 FAISS 인덱스를 로드합니다.
    # allow_dangerous_deserialization=True 옵션은 역직렬화를 허용합니다.
    load_db = FAISS.load_local(
        f'{current_dir}/restaurant-faiss', embeddings, allow_dangerous_deserialization=True)

    # 검색할 쿼리 문자열을 정의합니다.
    query = "음식점의 룸 서비스는 어떻게 운영되나요?"

    # 로드된 FAISS 인덱스를 사용하여 쿼리와 유사한 문서를 검색합니다.
    # `similarity_search` 메서드는 쿼리와 가장 유사한 문서를 찾기 위해 사용됩니다.
    # `query` 변수는 사용자가 검색하려는 질문이나 문장을 담고 있습니다.
    # `k=2`는 가장 유사한 문서 2개를 반환하도록 지정합니다.
    result = load_db.similarity_search(query, k=2)

    # 검색 결과를 출력합니다.
    print(result, "\n")

    # 쿼리를 임베딩 벡터로 변환합니다.
    embedding_vector_query = embeddings.embed_query(query)

    # 임베딩 벡터를 사용하여 비동기 방식으로 유사한 문서를 검색합니다.
    docs = await load_db.asimilarity_search_by_vector(embedding_vector_query)

    # 검색된 문서 중 첫 번째 문서를 출력합니다.
    print(docs[0])

# 이 스크립트가 직접 실행될 때, main() 함수를 asyncio.run()을 사용하여 실행합니다.
if __name__ == "__main__":
    asyncio.run(main())
