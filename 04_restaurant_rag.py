from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# pip install faiss-cpu or faiss-gpu
# 현재 파이썬 스크립트 실행 위치 반환
current_dir = os.path.dirname(os.path.abspath(__file__))
# 현재 파이썬 스크립트 실행 같은 위치에 있는 "restaurant-faiss" 폴더 경로
restaurant_faiss = os.path.join(current_dir, "restaurant-faiss")

# TextLoader 클래스를 사용하여 "restaurant.txt"라는 파일에서 텍스트를 로드합니다.
# 윈도우 사용자는 경로 문제시 "loader = TextLoader(f'{current_dir}\\restaurant.txt', encoding='utf-8')" 이 문구로 사용하시면 됩니다.
loader = TextLoader(f'{current_dir}/restaurants.txt')

# 파일의 내용을 document 객체로 로드합니다.
documents = loader.load()

# 텍스트를 300자 단위로 나누고, 연속된 청크 사이에 50자의 겹침을 두어 텍스트를 분할하는 text splitter 객체를 생성합니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 로드된 문서를 지정된 크기와 겹침에 따라 더 작은 청크로 분할합니다.
docs = text_splitter.split_documents(documents)

# 환경 변수에서 가져온 OpenAI API 키를 사용하여 OpenAIEmbeddings 클래스를 초기화합니다.
# 이 클래스는 텍스트 청크를 임베딩(숫자 표현)으로 변환하는 데 사용됩니다.
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# 문서 청크 목록과 해당 임베딩을 사용하여 FAISS 인덱스를 생성합니다.
# FAISS는 밀집 벡터의 효율적인 유사성 검색과 클러스터링을 위한 라이브러리입니다.
db = FAISS.from_documents(docs, embeddings)

# 생성된 FAISS 인덱스를 나중에 사용할 수 있도록 "restaurant-faiss"라는 로컬 디렉토리에 저장합니다.
db.save_local(restaurant_faiss)
print("Restaurant embedding index saved to", restaurant_faiss)
