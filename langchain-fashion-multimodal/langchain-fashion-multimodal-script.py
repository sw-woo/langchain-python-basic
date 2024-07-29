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
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# OpenAI API 키 설정 (환경 변수 사용 권장)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 스크립트 파일의 디렉토리 경로를 가져옵니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_dataset():
    dataset = load_dataset("detection-datasets/fashionpedia")
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    return dataset, dataset_folder


def save_images(dataset, dataset_folder, num_images=500):
    for i in range(num_images):
        image = dataset['train'][i]['image']
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))
    print(f"Saved {num_images} images to {dataset_folder}")


def setup_chroma_db():
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    chroma_client = chromadb.PersistentClient(path=vdb_path)
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(
        name="image", embedding_function=CLIP, data_loader=image_loader)
    return image_vdb


def add_images_to_db(image_vdb, dataset_folder):
    ids = []
    uris = []
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            file_path = os.path.join(dataset_folder, filename)
            ids.append(str(i))
            uris.append(file_path)
    image_vdb.add(ids=ids, uris=uris)
    print("Images added to the database.")


def query_db(image_vdb, query, results=3):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])


def print_results(results):
    for idx, uri in enumerate(results['uris'][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        print("\n")


def translate(text, target_lang):
    translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator. Translate the following text to {
         target_lang}."),
        ("user", "{text}")
    ])
    translation_chain = translation_prompt | translation_model | StrOutputParser()
    return translation_chain.invoke({"text": text})


def setup_vision_chain():
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
    return image_prompt | gpt4 | parser


def format_prompt_inputs(data, user_query):
    inputs = {}

    # Add user query to the dictionary
    inputs['user_query'] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

    # Encode the first image
    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    # Encode the second image
    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return inputs


def main():
    # Check if the dataset folder and images exist
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
        dataset, dataset_folder = setup_dataset()
        save_images(dataset, dataset_folder)
    else:
        print("Dataset folder and images already exist, skipping dataset setup.")

    # Check if the vector database is already set up
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
        image_vdb = setup_chroma_db()
        add_images_to_db(image_vdb, dataset_folder)
    else:
        print("Vector database already set up, skipping database setup.")
        image_vdb = setup_chroma_db()

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


if __name__ == "__main__":
    main()
