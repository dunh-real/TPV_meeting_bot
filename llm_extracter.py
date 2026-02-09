import os
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from clean_text import TextCleaner
from llm_service import get_llm_model

# CẤU HÌNH
BASE_SYSTEM_PROMPT = """
Bạn là một trợ lý AI thông minh chuyên phân tích nội dung cuộc họp/hội thoại.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên nội dung transcript được cung cấp.
Nếu thông tin không có trong transcript, hãy nói "Thông tin này không có trong tài liệu".
"""

def main():
    print("⏳ Đang khởi tạo dịch vụ LLM...")
    llm = get_llm_model()

    cleaner = TextCleaner()

    while True:
        print("\n" + "="*100)
        print("📂 NHẬP FILE TRANSCRIPT")
        print("(Gõ 'exit' hoặc 'quit' để thoát chương trình hoàn toàn)")
        print("="*100)
        
        file_path_input = input("👉 Nhập đường dẫn file .txt: ").strip()

        # Xử lý input đường dẫn (bỏ dấu ngoặc kép nếu user copy path dạng "C:\...")
        file_path = file_path_input.replace('"', '').replace("'", "")

        if file_path.lower() in ["exit", "quit"]:
            print("Tạm biệt!")
            sys.exit()

        # xử lý file và làm sạch
        if not os.path.exists(file_path):
            print("❌ File không tồn tại! Vui lòng nhập lại.")
            continue

        print("⏳ Đang làm sạch và đọc file...")
        try:
            # Lấy nội dung mới
            context_content = cleaner.clean_and_overwrite(file_path)
            print("✅ Đã nạp dữ liệu thành công!")
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            continue

        # SYSTEM PROMPT MỚI
        # nội dung cũ bị xóa sạch, chỉ ghép Base Prompt + Nội dung file mới
        full_system_message = f"""
        {BASE_SYSTEM_PROMPT}
        === NỘI DUNG TRANSCRIPT BẠN CẦN DÙNG ĐỂ TRẢ LỜI CÁC THÔNG TIN LIÊN QUAN ===
        {context_content}
        """
                
        # Reset lịch sử chat (Xóa ký ức về file cũ)
        chat_history = [SystemMessage(content=full_system_message)]

        print("\n🤖 BẮT ĐẦU CHAT VỚI FILE MỚI")
        print("(Gõ 'new' để đổi file khác, 'exit' hoặc 'quit' để thoát chương trình)")
        print("=" * 100)

        # CHAT
        while True:
            user_input = input("\n👤 Bạn: ").strip()

            # Kiểm tra lệnh điều hướng
            if user_input.lower() in ["exit", "quit"]:
                print("Tạm biệt!")
                sys.exit()
            
            if user_input.lower() in ["new"]:
                print("🔄 Đang chuyển sang nhập file mới...")
                break # Thoát vòng lặp chat để quay lại vòng lặp nhập file

            if not user_input:
                continue

            # Thêm câu hỏi vào lịch sử
            chat_history.append(HumanMessage(content=user_input))

            try:
                # Gọi LLM
                response = llm.invoke(chat_history)
                
                print(f"🧠 Bot: {response.content}")

                # Lưu câu trả lời vào lịch sử
                chat_history.append(AIMessage(content=response.content))

            except Exception as e:
                print(f"❌ Lỗi khi gọi API: {e}")

if __name__ == "__main__":
    main()