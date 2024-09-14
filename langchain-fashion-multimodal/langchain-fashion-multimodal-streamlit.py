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

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Get the script directory
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
    clip = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(
        name="image", embedding_function=clip, data_loader=image_loader)
    return image_vdb


def get_existing_ids(image_vdb):
    existing_ids = set()
    try:
        # Assuming 'list' method or similar functionality exists to get all IDs
        records = image_vdb.query(
            query_texts=[""], n_results=1000, include=["ids"])
        for record in records["ids"]:
            existing_ids.update(record)
    except Exception as e:
        print(f"Error retrieving existing IDs: {e}")
    return existing_ids


def add_images_to_db(image_vdb, dataset_folder):
    existing_ids = get_existing_ids(image_vdb)
    ids = []
    uris = []
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            img_id = str(i)
            if img_id not in existing_ids:
                file_path = os.path.join(dataset_folder, filename)
                ids.append(img_id)
                uris.append(file_path)
    if ids:
        image_vdb.add(ids=ids, uris=uris)
        print("New images added to the database.")
    else:
        print("No new images to add to the database.")


def query_db(image_vdb, query, results=3):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])


def translate(text, target_lang):
    translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator. Translate the following text to{target_lang}."),
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
    inputs = {'user_query': user_query}
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return inputs


def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Main function to run the Streamlit app


def main():
    st.set_page_config(page_title="FashionRAG", layout="wide")
    st.title("FashionRAG: Fashion and Styling Assistant")

    # Check if the dataset folder and images exist
    dataset_folder = os.path.join(SCRIPT_DIR, 'fashion_dataset')
    if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
        with st.spinner("Setting up dataset and saving images..."):
            dataset, dataset_folder = setup_dataset()
            save_images(dataset, dataset_folder)
        st.success("Dataset setup and images saved.")
    else:
        st.info("Dataset folder and images already exist, skipping dataset setup.")

    # Check if the vector database is already set up
    vdb_path = os.path.join(SCRIPT_DIR, 'img_vdb')
    if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
        with st.spinner("Setting up vector database and adding images..."):
            image_vdb = setup_chroma_db()
            add_images_to_db(image_vdb, dataset_folder)
        st.success("Vector database setup and images added.")
    else:
        st.info("Vector database already set up, skipping database setup.")
        image_vdb = setup_chroma_db()

    vision_chain = setup_vision_chain()

    st.header("Ask for styling advice")

    query_ko = st.text_input("Enter your styling question (in Korean):")

    if query_ko:
        with st.spinner("Translating and querying..."):
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
