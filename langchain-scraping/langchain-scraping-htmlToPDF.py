import os
from datetime import datetime
import pytz
import asyncio

from playwright.async_api import async_playwright
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# playwright를 사용하여 PDF 파일로 크롤링하는 부분


async def generate_pdf(url, output_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(5000)
            await page.pdf(path=output_path, format="A4")
            print(f"PDF successfully generated and saved to {output_path}")
        except Exception as e:
            print(f"Error during PDF generation:{e}")
        finally:
            await browser.close()


def get_current_time():
    seoul_tz = pytz.timezone("Asia/Seoul")
    return datetime.now(seoul_tz).strftime("%Y-%M-%d %H:%M:%S")


# 저장할 PDF 파일 위치 선언
url = "https://news.naver.com/section/101"
pdf_dir = os.path.join(os.getcwd(), "pdf")
time_file_name = f"{get_current_time()}-save.pdf"

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

os.makedirs(pdf_dir, exist_ok=True)


async def main():
    # PDF 생성
    pdf_path = os.path.join(pdf_dir, time_file_name)
    print(f"Attempting to save PDF to : {pdf_path}")
    try:
        await generate_pdf(url, pdf_path)
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"PDF file created. Size: {file_size} bytes")
            # PDF 로딩 및 처리
            if file_size > 0:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                print(f"PDF loader 확인 부분: {docs}")
            else:
                print("PDF file is empty")
        else:
            print("PDF file was not created")

    except Exception as e:
        print(f"Error generation PDF:{e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
