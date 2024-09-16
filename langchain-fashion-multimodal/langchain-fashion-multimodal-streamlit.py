import os
import base64
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키가 설정되지 않았을 경우 에러 발생
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 스크립트 파일의 디렉토리 경로 가져오기
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋을 설정하는 함수
def setup_dataset():
    # 패션 관련 데이터셋 불러오기
    dataset = load_dataset("detection-datasets/fashionpedia")
    # 데이터셋을 저장할 폴더 경로 설정
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    # 폴더가 없으면 생성
    os.makedirs(dataset_folder, exist_ok=True)
    return dataset, dataset_folder

# 데이터셋에서 이미지를 저장하는 함수
def save_images(dataset, dataset_folder, num_images=1000):
    # 주어진 수의 이미지를 저장
    for i in range(num_images):
        image = dataset['train'][i]['image']
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))
    print(f"{num_images}개의 이미지를 {dataset_folder}에 저장했습니다.")

# Chroma 데이터베이스를 설정하는 함수
def setup_chroma_db():
    # 벡터 데이터베이스 저장 경로 설정
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    # Chroma 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=vdb_path)
    # 이미지 로더 및 OpenCLIP 임베딩 함수 설정
    image_loader = ImageLoader()
    clip = OpenCLIPEmbeddingFunction()
    # 이미지 데이터베이스 생성 또는 가져오기
    image_vdb = chroma_client.get_or_create_collection(
        name="image", embedding_function=clip, data_loader=image_loader)
    return image_vdb

# 기존에 존재하는 이미지 IDs를 가져오는 함수
def get_existing_ids(image_vdb,dataset_folder):
    existing_ids = set()
    try:
        # dataset_folder 내의 이미지 파일 수 계산
        num_images = len([name for name in os.listdir(dataset_folder)])
        print(f"데이터 폴더 전체 이미지수:{num_images}")

        records = image_vdb.query(
            query_texts=[""], n_results=num_images, include=["ids"])
        for record in records["ids"]:
            existing_ids.update(record)
            print(f"{len(record)} 존재 IDs")
    except Exception as e:
        print(f"{len(record)}개의 기존 IDs가 있습니다.")
    return existing_ids

# 이미지를 데이터베이스에 추가하는 함수
def add_images_to_db(image_vdb, dataset_folder):
    existing_ids = get_existing_ids(image_vdb,dataset_folder)
    ids = []
    uris = []
    # 폴더에서 이미지를 읽어와서 데이터베이스에 추가
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            img_id = str(i)
            if img_id not in existing_ids:
                file_path = os.path.join(dataset_folder, filename)
                ids.append(img_id)
                uris.append(file_path)
    if ids:
        image_vdb.add(ids=ids, uris=uris)
        print("새로운 이미지를 데이터베이스에 추가했습니다.")
    else:
        print("추가할 새로운 이미지가 없습니다.")

# 데이터베이스에서 쿼리를 실행하는 함수
def query_db(image_vdb, query, results=2):
    # 주어진 쿼리를 실행하고, 상위 결과 반환
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])

# 텍스트를 지정된 언어로 번역하는 함수
def translate(text, target_lang):
    # OpenAI의 ChatGPT 모델을 사용하여 번역
    translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    # 번역에 사용할 프롬프트 생성
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator. Translate the following text to{target_lang}."),
        ("user", "{text}")
    ])
    # 번역 체인 설정
    translation_chain = translation_prompt | translation_model | StrOutputParser()
    # 번역 결과 반환
    return translation_chain.invoke({"text": text})

# 시각적 정보를 처리하는 체인을 설정하는 함수
def setup_vision_chain():
    # GPT-4 모델을 사용하여 시각적 정보를 처리 gpt-4o or gpt-4o-mini 모델선택
    gpt4 = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = StrOutputParser()
    image_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful fashion and styling assistant. Answer the user's question using the given image context with direct references to parts of the images provided. Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure."),
        ("user", [
            {"type": "text", "text": "What are some ideas for styling {user_query}"},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
        ]),
    ])
    # 프롬프트, 모델, 파서 체인을 반환
    return image_prompt | gpt4 | parser

# 프롬프트 입력을 포맷하는 함수
def format_prompt_inputs(data, user_query):
    inputs = {}

    # 사용자 쿼리를 딕셔너리에 추가
    inputs['user_query'] = user_query
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return inputs

# 이미지를 Base64로 로드하는 함수
def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    

# Streamlit 앱을 실행하는 메인 함수
def main():
    st.set_page_config(page_title="FashionRAG", layout="wide")
    st.title("FashionRAG: 패션 및 스타일링 어시스턴트")

    # 데이터셋 폴더 및 이미지 존재 여부 확인
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
        with st.spinner("데이터셋 설정 및 이미지 저장 중..."):
            dataset, dataset_folder = setup_dataset()
            save_images(dataset, dataset_folder)
        st.success("데이터셋 설정 및 이미지 저장 중...")
    else:
        st.info("데이터셋이 설정되고 이미지가 저장되었습니다.")

    # 벡터 데이터베이스 설정 여부 확인
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
        with st.spinner("벡터 데이터베이스 설정 및 이미지 추가 중..."):
            image_vdb = setup_chroma_db()
            add_images_to_db(image_vdb, dataset_folder)
        st.success("벡터 데이터베이스 설정 및 이미지 추가가 완료되었습니다.")
    else:
        st.info("벡터 데이터베이스가 이미 설정되어 있습니다. 데이터베이스 설정을 건너뜁니다.")
        image_vdb = setup_chroma_db()

    vision_chain = setup_vision_chain()

    st.header("스타일링 조언을 받아보세요")

    query_ko = st.text_input("스타일링에 대한 질문을 입력하세요:")

    if query_ko:
        with st.spinner("번역 및 쿼리 진행 중..."):
            query_en = translate(query_ko, "English")
            results = query_db(image_vdb, query_en, results=2)
            prompt_input = format_prompt_inputs(results, query_en)
            response_en = vision_chain.invoke(prompt_input)
            response_ko = translate(response_en, "Korean")

        st.subheader("검색된 이미지:")
        for idx, uri in enumerate(results['uris'][0]):
            img_base64 = load_image_as_base64(uri)
            img_data_url = f"data:image/png;base64,{img_base64}"
            st.image(img_data_url, caption=f"ID: {results['ids'][0][idx]}", width=300)

        st.subheader("FashionRAG의 응답:")
        st.markdown(response_ko)


if __name__ == "__main__":
    main()
