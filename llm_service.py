import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm_model(model_name: str = "gemini-2.0-flash", temperature: float = 0.5):
    
    # 1. Lấy API Key
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ LỖI: Chưa tìm thấy 'API_KEY' trong file .env")
        return None

    # 2. Khởi tạo Model Gemini
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True # TRUE để tương thích tốt hơn nếu LangChain cũ
            
        )
        return llm
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Gemini: {e}")
        return None