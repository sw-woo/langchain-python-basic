import os
from dotenv import load_dotenv
import chromadb
import base64
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

# API 키가 설정되지 않았을 경우 에러 설정
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

# OpenAI API 키 설정 (환경 변수 사용 권장)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 스크립트 파일의 디렉토리 경로를 가져옵니다.
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
def save_images(dataset, dataset_folder, num_images=500):
    # 주어진 수의 이미지를 저장
    for i in range(num_images):
        image = dataset['train'][i]['image']
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))
    print(f"Saved {num_images} images to {dataset_folder}")

# Chroma 데이터베이스를 설정하는 함수
def setup_chroma_db():
    # 벡터 데이터베이스 저장 경로 설정
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    # Chroma 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=vdb_path)
    # 이미지 로더 및 OpenCLIP 임베딩 함수 설정
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    # 이미지 데이터베이스 생성 또는 가져오기
    image_vdb = chroma_client.get_or_create_collection(
        name="image", embedding_function=CLIP, data_loader=image_loader)
    return image_vdb

# 이미지를 데이터베이스에 추가하는 함수
def add_images_to_db(image_vdb, dataset_folder):
    ids = []
    uris = []
    # 폴더에서 이미지를 읽어와서 데이터베이스에 추가
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            file_path = os.path.join(dataset_folder, filename)
            ids.append(str(i))
            uris.append(file_path)
    image_vdb.add(ids=ids, uris=uris)
    print("이미지가 데이터베이스에 추가되었습니다.")

# 데이터베이스에서 쿼리를 실행하는 함수
def query_db(image_vdb, query, results=2):
    # 주어진 쿼리를 실행하고, 상위 결과 반환
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])

# 결과를 출력하는 함수
def print_results(results):
    for idx, uri in enumerate(results['uris'][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        print("\n")

# 텍스트를 지정된 언어로 번역하는 함수
def translate(text, target_lang):
    # OpenAI의 ChatGPT 모델을 사용하여 번역
    translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    # 번역에 사용할 프롬프트 생성
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator. Translate the following text to {target_lang}."),
        ("user", "{text}")
    ])
    # 번역 체인 설정
    translation_chain = translation_prompt | translation_model | StrOutputParser()
    # 번역 결과 반환
    return translation_chain.invoke({"text": text})

# 시각적 정보를 처리하는 체인을 설정하는 함수
def setup_vision_chain():
    # GPT-4 모델을 사용하여 시각적 정보를 처리 gpt-4o or gpt-4o-mini 모델선택
    gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
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

    # 'uris' 리스트에서 첫 두 이미지 경로 가져오기
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

     # 첫 번째 이미지 인코딩
    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    # 두 번째 이미지 인코딩
    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return inputs

# 메인 함수
def main():
    # 데이터셋 폴더 및 이미지가 있는지 확인
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
        dataset, dataset_folder = setup_dataset()
        save_images(dataset, dataset_folder)
    else:
        print("데이터셋 폴더와 이미지가 이미 존재합니다. 데이터셋 설정을 건너뜁니다.")

    # 벡터 데이터베이스가 설정되었는지 확인
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
        image_vdb = setup_chroma_db()
        add_images_to_db(image_vdb, dataset_folder)
    else:
        print("벡터 데이터베이스가 이미 설정되어 있습니다. 데이터베이스 설정을 건너뜁니다.")
        image_vdb = setup_chroma_db()

    #시각적 정보를 처리하는 체인 설정
    vision_chain = setup_vision_chain() 

    while True:
        print("\nFashionRAG가 여러분의 서비스를 위해 준비되었습니다!")
        print("오늘 어떤 스타일에 대해 조언을 받고 싶으신가요? (종료하려면 'quit' 입력)")
        query_ko = input("\n질문을 입력하세요: ")

        if query_ko.lower() == 'quit':
            break

        query_en = translate(query_ko, "English")
        results = query_db(image_vdb, query_en, results=2)
        prompt_input = format_prompt_inputs(results, query_en)
        response_en = vision_chain.invoke(prompt_input)
        response_ko = translate(response_en, "Korean")

        print("\n검색된 이미지:")
        print_results(results)

        print("\nFashionRAG의 응답:")
        print(response_ko)

# 메인 함수 실행
if __name__ == "__main__":
    main()
