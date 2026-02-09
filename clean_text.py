import re
import os

class TextCleaner:

    @staticmethod
    def clean_and_overwrite(file_path: str) -> str:
        
        # Đọc file
        if not os.path.exists(file_path):
            return "File không tồn tại."

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        merged_conversation = []
        
        # biến trạng thái
        current_speaker = None
        current_text_buffer = []
        current_start_time = None

        # Regex để bắt format: [Start - End] SPEAKER_XX: Content
        pattern = re.compile(r"\[(.*?)\] (SPEAKER_\d+):(.*)")

        for line in lines:
            line = line.strip()
            if not line: continue

            match = pattern.match(line)
            
            # 1: Dòng có chứa Timestamp và Speaker
            if match:
                timestamp_raw, speaker, content = match.groups()
                content = content.strip()
                
                # Nếu vẫn là người cũ đang nói -> Gộp vào buffer
                if speaker == current_speaker:
                    current_text_buffer.append(content)
                
                # Nếu đổi người nói mới
                else:
                    # Lưu đoạn hội thoại của người trước đó (nếu có)
                    if current_speaker:
                        full_text = " ".join(current_text_buffer)
                        merged_conversation.append(f"[{current_start_time}] {current_speaker}: {full_text}")

                    # Reset trạng thái cho người mới
                    current_speaker = speaker
                    current_text_buffer = [content]
                    # Lấy thời gian bắt đầu (trước dấu -)
                    current_start_time = timestamp_raw.split('-')[0].strip()

            # 2: Dòng không có Timestamp (nội dung bị xuống dòng)
            # nối tiếp vào người đang nói hiện tại
            else:
                if current_speaker:
                    current_text_buffer.append(line)

        # Lưu đoạn cuối cùng sau khi hết vòng lặp
        if current_speaker and current_text_buffer:
            full_text = " ".join(current_text_buffer)
            merged_conversation.append(f"[{current_start_time}] {current_speaker}: {full_text}")

        # Kết quả cuối cùng
        final_content = "\n\n".join(merged_conversation)

        # 2. GHI ĐÈ VÀO FILE GỐC
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        return final_content

if __name__ == "__main__":
    cleaner = TextCleaner()
    
    file_path = r"C:\AI Project\Chatbot recording\Text\transcription2.txt"
    
    try:
        result = cleaner.clean_and_overwrite(file_path)
        print("--- ĐÃ XỬ LÝ VÀ GHI ĐÈ THÀNH CÔNG ---")
        print(result[:500])
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")